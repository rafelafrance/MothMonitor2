#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

from moth.external.engine import evaluate, train_one_epoch
from moth.pylib import bbox, moth_dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD_DEV = [0.229, 0.224, 0.225]


def train_action(args: argparse.Namespace) -> None:
    stats_split = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, bbox.BBOX_NUM_CLASSES
    )

    model.to(device)

    train_xforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.5, hue=0.3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD_DEV),
        ]
    )
    valid_xforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD_DEV),
        ]
    )

    train_dataset = moth_dataset.MothDataset(
        bbox_json=args.train_json, transforms=train_xforms, limit=args.limit
    )
    valid_dataset = moth_dataset.MothDataset(
        bbox_json=args.valid_json, transforms=valid_xforms, limit=args.limit
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    best_precision = 0.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)

        coco_eval = evaluate(model, valid_loader, device=device)

        precision = sum(coco_eval.coco_eval["bbox"].stats[:stats_split]) / stats_split
        recall = sum(coco_eval.coco_eval["bbox"].stats[stats_split:]) / stats_split

        if args.best_checkpoint and precision > best_precision:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "precision": precision,
                "recall": recall,
            }
            torch.save(state, args.best_checkpoint)

        best_precision = max(precision, best_precision)

        print(
            f"Precision {precision} Recall {recall} Best precision {best_precision}\n"
        )


def score_action(args: argparse.Namespace) -> None:
    pass


def inference_action(args: argparse.Namespace) -> None:
    pass


def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Edit data in the bug bounding boxes JSON file.
            Backup the JSON file you're going to edit first.
            """
        ),
    )

    subparsers = arg_parser.add_subparsers(
        title="Subcommands", description="Train, score, or infer a model."
    )

    # ------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="""Train a model. This function expects that the training and
            validation images are already correctly sized and formatted.
            It will perform image augmentations for training.""",
    )

    # train_parser.add_argument(
    #     "--model-name",
    #     default="IDEA-Research/dab-detr-resnet-50",
    #     metavar="PATH",
    #     help="""Finetune this model. (default: %(default)s)""",
    # )

    train_parser.add_argument(
        "--best-checkpoint",
        type=Path,
        metavar="PATH",
        help="""Save the best model checkpoint to this file.""",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="INT",
        help="""How many epochs to train. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Images are in this directory.""",
    )

    train_parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""JSON file containing training data.""",
    )

    train_parser.add_argument(
        "--valid-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""JSON file containing validation data.""",
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="INT",
        help="""How many images to process at a time. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=3e-4,
        metavar="FLOAT",
        help="""Initial learning rate for the optimizer. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        metavar="FLOAT",
        help="""Weight decay for the optimizer. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="""Use the asmgrad variant of the optimizer.""",
    )

    train_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the dataset to this many records. Used for testing mostly.""",
    )

    train_parser.set_defaults(func=train_action)

    # ------------------------------------------------------------
    score_parser = subparsers.add_parser(
        "score",
        help="""Score a trained model. This function expects that the
            test/score/hold out dataset is aready correctly sized and formatted.""",
    )

    score_parser.add_argument(
        "--model-name",
        default="IDEA-Research/dab-detr-resnet-50",
        metavar="PATH",
        help="""Finetune this model. (default: %(default)s)""",
    )

    score_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Score the checkpoint in this directory.""",
    )

    score_parser.add_argument(
        "--test-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""JSON file containing test data.""",
    )

    score_parser.add_argument(
        "--image-width",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this wide (pixels) for the model. (default: %(default)s)""",
    )

    score_parser.add_argument(
        "--image-height",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this high (pixels) for the model. (default: %(default)s)""",
    )

    score_parser.set_defaults(func=score_action)

    # ------------------------------------------------------------
    inference_parser = subparsers.add_parser(
        "inference",
        help="""Run inference on a directory of images.""",
    )

    inference_parser.add_argument(
        "--model-name",
        default="IDEA-Research/dab-detr-resnet-50",
        metavar="PATH",
        help="""Base model. (default: %(default)s)""",
    )

    inference_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Run inference using this model checkpoint.""",
    )

    inference_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory of images to infer.""",
    )

    inference_parser.add_argument(
        "--image-width",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this wide (pixels) for the model. (default: %(default)s)""",
    )

    inference_parser.add_argument(
        "--image-height",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this high (pixels) for the model. (default: %(default)s)""",
    )

    inference_parser.set_defaults(func=inference_action)

    # ------------------------------------------------------------
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

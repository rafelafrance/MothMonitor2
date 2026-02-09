#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

from moth.external.coco_eval import COCOeval
from moth.external.engine import evaluate, train_one_epoch
from moth.pylib import bbox, moth_dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD_DEV = [0.229, 0.224, 0.225]


def train_action(args: argparse.Namespace) -> None:
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
            v2.AutoAugment(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    valid_xforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_dataset = moth_dataset.MothDataset(
        bbox_json=args.train_json, transforms=train_xforms, limit=args.limit
    )
    valid_dataset = moth_dataset.MothDataset(
        bbox_json=args.valid_json, transforms=valid_xforms, limit=args.limit
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    # optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)

    best_precision = 0.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)

        # lr_scheduler.step()

        coco_eval = evaluate(model, valid_loader, device=device)
        precision, recall = get_stats(coco_eval)

        best_precision = save_state(
            epoch,
            model,
            optimizer,
            args.checkpoint_dir,
            precision,
            recall,
            best_precision,
        )


def get_stats(coco_eval: COCOeval) -> tuple[float, float]:
    stats_split = 6
    precision = sum(coco_eval.coco_eval["bbox"].stats[:stats_split]) / stats_split
    recall = sum(coco_eval.coco_eval["bbox"].stats[stats_split:]) / stats_split
    return precision, recall


def save_state(
    epoch: int,
    model: FasterRCNN,
    optimizer: Optimizer,
    checkpoint_dir: Path,
    precision: float,
    recall: float,
    best_precision: float,
) -> float:
    best_precision = max(precision, best_precision)

    if checkpoint_dir and precision > best_precision:
        state_path = checkpoint_dir / f"checkpoint_{epoch}.pth"
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "precision": precision,
            "recall": recall,
        }
        torch.save(state, state_path)

    return best_precision


# def train_one_epoch(
#     model: FasterRCNN,
#     device: torch.device,
#     loader: DataLoader,
#     optimizer: Optimizer,
# ) -> float:
#     running_loss = 0.0
#
#     for images, targets in loader:
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         loss_dict = model(images, targets)
#
#         losses = sum(loss for loss in loss_dict.values())
#
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#
#         running_loss += losses.item()
#
#     return running_loss / len(loader)
#
#
# def score_one_epoch(
#     model: FasterRCNN,
#     device: torch.device,
#     transforms: v2.Compose,
#     coco: COCO,
#     image_dir: Path,
# ) -> float:
#     predictions = []
#
#     for image_id in coco.getImgIds():
#         image_rec = coco.loadImgs(image_id)[0]
#         image_path = str(image_dir / image_rec["file_name"])
#         image = Image.open(image_path)
#         image = transforms(image)
#         image /= 255.0
#         image = image.unsqueeze(0)
#         image = image.to(device)
#
#         with torch.no_grad():
#             outputs = model(image)
#
#         for output in outputs:
#             boxes = output["boxes"].cpu().numpy()
#             scores = output["scores"].cpu().numpy()
#             labels = output["labels"].cpu().numpy()
#
#             for box, score, label in zip(boxes, scores, labels, strict=True):
#                 prediction = {
#                     "image_id": int(image_id),
#                     "category_id": label,
#                     "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
#                     "score": score,
#                 }
#                 predictions.append(str(prediction))
#
#     pp(predictions[:10])
#     print([type(p["image_id"]) for p in predictions])
#     coco_results = coco.loadRes(predictions)
#     pp(coco_results.dataset["annotations"])
#     coco_eval = COCOeval(coco, coco_results, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#     print()
#
#     # return running_loss / len(loader)
#     return 0.0


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
        "--checkpoint-dir",
        type=Path,
        metavar="PATH",
        help="""Save model checkpoints in this directory.""",
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

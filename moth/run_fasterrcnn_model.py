#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models import detection
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2
from tqdm import tqdm

from moth.pylib import bbox
from moth.pylib.moth_dataset import MothDataset

OPTIMIZER = torch.optim.SGD | torch.optim.Adam | torch.optim.AdamW

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD_DEV = [0.229, 0.224, 0.225]


def train_action(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes + 1)

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

    train_dataset = MothDataset(
        bbox_json=args.train_json,
        transforms=train_xforms,
        num_classes=args.num_classes,
        limit=args.limit,
    )
    valid_dataset = MothDataset(
        bbox_json=args.valid_json, num_classes=args.num_classes, limit=args.limit
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    prev_state = {}
    if args.previous_checkpoint:
        prev_state = torch.load(args.previous_checkpoint, weights_only=False)
        model.load_state_dict(prev_state["state_dict"])

    best_epoch = prev_state.get("epoch", 0)
    start_epoch = best_epoch + 1
    best_map05 = prev_state.get("map@0.5", 0.0)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_class = prev_state.get("optimizer_class", args.optimizer_class)
    match optimizer_class.lower():
        case "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        case "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=args.amsgrad,
            )
        case "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=args.amsgrad,
            )
    if optimizer_state := prev_state.get("optimizer"):
        optimizer.load_state_dict(optimizer_state)

    valid_coco = valid_dataset.to_coco_obj()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Training loop
        running_loss = 0.0
        model.train()
        for images, targets in tqdm(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        # Validation loop
        map05 = score_data(model, valid_coco, args.image_dir, device)

        running_loss /= len(train_loader)

        if args.save_checkpoint and map05 > best_map05:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "map@0.5": map05,
                "optimizer_class": optimizer_class,
            }
            torch.save(state, args.save_checkpoint)

        flag = ""
        if map05 > best_map05:
            flag = "*"
            best_epoch = epoch

        best_map05 = max(best_map05, map05)

        print(
            f"Epoch {epoch} Loss: {running_loss:0.3f} mAP@0.5: {map05:0.3f} {flag}"
            f"\t\tBest Epoch {best_epoch} Best mAP@0.5 {best_map05:0.3f}\n"
        )


def score_action(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, bbox.BBOX_NUM_CLASSES
    )

    model.to(device)

    score_dataset = MothDataset(bbox_json=args.score_json, limit=args.limit)

    score_coco = score_dataset.to_coco_obj()

    map05 = score_data(model, score_coco, args.image_dir, device)

    print(f"Score mAP@0.5 = {map05:0.3f}")
    print()


def score_data(
    model: detection, coco_dataset: COCO, image_dir: Path, device: torch.device
) -> float:
    predictions = []
    model.eval()
    for image_id in tqdm(coco_dataset.getImgIds()):
        image_rec = coco_dataset.loadImgs(image_id)[0]
        image = Image.open(image_dir / image_rec["file_name"])
        image = v2.PILToTensor()(image).float() / 255.0
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            preds = model(image)
        for pred in preds:
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                prediction = {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": score,
                }
                predictions.append(prediction)

    map05 = 0.0
    if predictions:
        results = coco_dataset.loadRes(predictions)
        coco_eval = COCOeval(coco_dataset, results, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        __, map05, *_ = coco_eval.stats

    return map05


def inference_action(args: argparse.Namespace) -> None:
    pass


def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Train, score, or infer a COCO model."""),
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
    train_parser.add_argument(
        "--save-checkpoint",
        type=Path,
        metavar="PATH",
        help="""Save the best model checkpoint to this file.""",
    )
    train_parser.add_argument(
        "--previous-checkpoint",
        type=Path,
        metavar="PATH",
        help="""Continue training this this checkpoint.""",
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
        "--optimizer-class",
        choices=["sgd", "adam", "adamw"],
        default="sgd",
        help="""Which optimizer to use. (default: %(default)s)""",
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
        default=1e-3,
        metavar="FLOAT",
        help="""Weight decay for the optimizer. (default: %(default)s)""",
    )
    train_parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="FLOAT",
        help="""Optimizer momentum for SGD. (default: %(default)s)""",
    )
    train_parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="""Use the asmgrad variant of the optimizer.""",
    )
    train_parser.add_argument(
        "--num-classes",
        type=int,
        default=bbox.BBOX_NUM_CLASSES,
        metavar="INT",
        help="""Train on this many classes. Used to lower the number of classes.
        (default: %(default)s)""",
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

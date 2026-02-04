#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import moth.pylib.bbox

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD_DEV = [0.229, 0.224, 0.225]


def train_action(args: argparse.Namespace) -> None:
    model = torchvision.models.detection.faster_rcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      moth.pylib.bbox.NUM_CLASSES)


def score_action(args: argparse.Namespace) -> None:
    pass


def inference_action(args: argparse.Namespace) -> None:
    pass


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

    train_parser.add_argument(
        "--model-name",
        default="IDEA-Research/dab-detr-resnet-50",
        metavar="PATH",
        help="""Finetune this model. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--output-dir",
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
        "--image-width",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this wide (pixels) for the model. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--image-height",
        type=int,
        metavar="INT",
        default=480,
        help="""Images are this high (pixels) for the model. (default: %(default)s)""",
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

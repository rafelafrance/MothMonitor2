#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path


def train_action(args: argparse.Namespace) -> None:
    pass


def score_action(args: argparse.Namespace) -> None:
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
        title="Subcommands", description="Train, test, or infer a model."
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
        metavar="MODEL",
        help="""Which model are we using. (default: %(default)s)""",
    )

    train_parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""JSON file containing training data.""",
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

    train_parser.set_defaults(func=train_action)

    # ------------------------------------------------------------
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

#!/usr/bin/env python3

import argparse
import glob
import json
import random
import textwrap
from pathlib import Path

import imagesize
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


def moved_action(args: argparse.Namespace) -> None:
    with args.bbox_json.open() as f:
        images = json.load(f)

    for image in images:
        old = Path(image["path"])
        new = args.new_dir / old.name
        image["path"] = str(new)

    with args.bbox_json.open("w") as f:
        json.dump(images, f, indent=4)


def sample_action(args: argparse.Namespace) -> None:
    args.sample_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for path in glob.glob(args.dir_glob):
        path = Path(path)
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            images.append(path)

    images = sorted(images)
    random.seed(args.seed)
    images = random.choices(images, k=args.samples)

    for path in tqdm(images):
        width, height = imagesize.get(path)
        width = int(width * args.scale)
        height = int(height * args.scale)

        image = Image.open(path)

        scaled = image.resize((width, height))

        if width > height:
            left = random.choice(range(width - args.long_edge))
            top = random.choice(range(height - args.short_edge))
            cropped = scaled.crop(
                (left, top, left + args.long_edge, top + args.short_edge)
            )
        else:
            left = random.choice(range(width - args.short_edge))
            top = random.choice(range(height - args.long_edge))
            cropped = scaled.crop(
                (left, top, left + args.short_edge, top + args.long_edge)
            )

        parent = str(path.parent).replace("/", "_")
        name = (
            args.sample_dir / f"{parent}_{path.stem}_left_{left}_top_{top}{path.suffix}"
        )
        cropped.save(name)


def count_action(args: argparse.Namespace) -> None:
    with args.bbox_json.open() as f:
        images = json.load(f)

    counts = {
        "empty": 0,
        "moth": 0,
        "not_moth": 0,
        "unsure": 0,
    }
    for image in images:
        if len(image["boxes"]) == 0:
            counts["empty"] += 1
        for content in ("moth", "not_moth", "unsure"):
            counts[content] += sum(1 for b in image["boxes"] if b["content"] == content)

    table = Table(title="Count Box Types")
    table.add_column("", no_wrap=True)
    table.add_column("Count", justify="right")

    table.add_row("Images with boxes", f"{len(images) - counts['empty']:,d}")
    table.add_row("Empty images", f"{counts['empty']:,d}")
    table.add_row("", "")
    table.add_row("Moth boxes", f"{counts['moth']:,d}")
    table.add_row("Not moth boxes", f"{counts['not_moth']:,d}")
    table.add_row("Unsure boxes", f"{counts['unsure']:,d}")

    console = Console()
    console.print(table)


def split_action(args: argparse.Namespace) -> None:
    with args.bbox_json.open() as f:
        all_images = json.load(f)

    all_images = sorted(all_images)
    random.seed(args.seed)
    all_images = random.shuffle(all_images)

    for id_, image in enumerate(all_images):
        image["id"] = id_

    total: int = len(all_images)
    split1: int = round(total * args.train_fract)
    split2: int = split1 + round(total * args.val_fract)

    image_splits: dict[str, list] = {
        "train": all_images[:split1],
        "valid": all_images[split1:split2],
        "test": all_images[split2:],
    }

    for split, images in image_splits.items():
        split = {"images": [], "annotations": []}
        for image in images:
            split["images"].append(
                {
                    "id": image["id"],
                    "width": image["width"],
                    "height": image["height"],
                    "image_path": image["path"],
                }
            )
            for box in image["boxes"]:
                bbox = {}


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
        title="Subcommands", description="Actions on bug bounding boxes"
    )

    # ------------------------------------------------------------
    moved_parser = subparsers.add_parser(
        "moved",
        help="""You moved the images to a new directory, possibly to a new computer,
            now update the bug bounding box JSON file to this new directory.""",
    )

    moved_parser.add_argument(
        "--bbox-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Edit this JSON file.""",
    )

    moved_parser.add_argument(
        "--new-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""You moved the images to this directory.""",
    )

    moved_parser.set_defaults(func=moved_action)

    # ------------------------------------------------------------
    sample_parser = subparsers.add_parser(
        "sample",
        help="""Sample images to create a datasets for training, validation,
            or testing. It samples image files, scales them, and cut out a random
            section (of the given size) and puts the images into a directory.""",
    )

    sample_parser.add_argument(
        "--dir-glob",
        type=str,
        required=True,
        metavar="GLOB",
        help="""Sample images from this set of directories. You need to quote this
            argument. Only image files are used (jpg, jpeg, png, etc.).
            For example --dir-glob 'data/images/**'.""",
    )

    sample_parser.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the sampled images to this directory.""",
    )

    sample_parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="""The raw images can be huge. You may want to scale them down (or up)
            so that they are easier to work with. (default: %(default)s)""",
    )

    sample_parser.add_argument(
        "--long-edge",
        type=int,
        default=1333,
        metavar="INT",
        help="""Randomly crop this size of an area out of the sampled image along the
            long edge of the image. (default: %(default)s)""",
    )

    sample_parser.add_argument(
        "--short-edge",
        type=int,
        default=800,
        metavar="INT",
        help="""Randomly crop this size of an area out of the sampled image along the
            short edge of the image. (default: %(default)s)""",
    )

    sample_parser.add_argument(
        "--samples",
        type=int,
        default=500,
        metavar="INT",
        help="""Number of samples to take. (default: %(default)s)""",
    )

    sample_parser.add_argument(
        "--seed",
        type=int,
        default=9885303,
        metavar="INT",
        help="""Use this as the random seed.""",
    )

    sample_parser.set_defaults(func=sample_action)

    # ------------------------------------------------------------
    count_parser = subparsers.add_parser(
        "count",
        help="""Count number of bounding boxes per image.""",
    )

    count_parser.add_argument(
        "--bbox-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Edit this JSON file.""",
    )

    count_parser.set_defaults(func=count_action)

    # ------------------------------------------------------------
    split_parser = subparsers.add_parser(
        "split",
        help="""Format and split images into test, valid, and test datasets.""",
    )

    split_parser.add_argument(
        "--bbox-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Use this JSON file as the input.""",
    )

    split_parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output to this directory.""",
    )

    split_parser.add_argument(
        "--train-fract",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="""What fraction of the records to use for training.
            (default: %(default)s)""",
    )

    split_parser.add_argument(
        "--test-fract",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="""What fraction of the records to use for test.
            (default: %(default)s)""",
    )

    split_parser.add_argument(
        "--seed",
        type=int,
        default=292583,
        help="""Seed for the random number generator. (default: %(default)s)""",
    )

    split_parser.set_defaults(func=split_action)
    # ------------------------------------------------------------

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

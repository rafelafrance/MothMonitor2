#!/usr/bin/env python3

import argparse
import glob
import json
import random
import textwrap
from pathlib import Path

import imagesize
from PIL import Image
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

        if width > height:
            left = random.choice(range(width - args.long_edge))
            top = random.choice(range(height - args.short_edge))
            cropped = image.crop(
                (left, top, left + args.long_edge, top + args.short_edge)
            )
        else:
            left = random.choice(range(width - args.short_edge))
            top = random.choice(range(height - args.long_edge))
            cropped = image.crop(
                (left, top, left + args.short_edge, top + args.long_edge)
            )

        parent = str(path.parent).replace("/", "_")
        name = (
            args.sample_dir / f"{parent}_{path.stem}_left_{left}_top_{top}{path.suffix}"
        )
        cropped.save(name)


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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

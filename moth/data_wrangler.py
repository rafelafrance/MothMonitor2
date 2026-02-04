#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from moth.pylib import bbox


def fix_action(args: argparse.Namespace) -> None:
    bbox_images = bbox.load_json(args.bbox_json)

    if args.new_dir:
        for image in bbox_images:
            old = Path(image.path)
            new = args.new_dir / old.name
            image.path = str(new)

    bbox.dump_json(bbox_images, args.bbox_json)


def count_action(args: argparse.Namespace) -> None:
    bbox_images = bbox.load_json(args.bbox_json)

    counts = {
        "empty": 0,
        "moth": 0,
        "not_moth": 0,
        "unsure": 0,
    }
    for image in bbox_images:
        if len(image.bboxes) == 0:
            counts["empty"] += 1
        for content in bbox.BBOX_FORMAT:
            counts[content] += sum(1 for b in image.bboxes if b.content == content)

    table = Table(title="Count Box Types")
    table.add_column("", no_wrap=True)
    table.add_column("Count", justify="right")

    table.add_row("Images with boxes", f"{len(bbox_images) - counts['empty']:,d}")
    table.add_row("Empty images", f"{counts['empty']:,d}")
    table.add_row("Total images", f"{len(bbox_images):,d}")
    table.add_row("", "")

    total = 0
    for content in bbox.BBOX_FORMAT:
        table.add_row(f"{content.title()} boxes", f"{counts[content]:,d}")
        total += counts[content]

    table.add_row("Total boxes", f"{total:,d}")

    console = Console()
    console.print(table)


def pics_action(args: argparse.Namespace) -> None:
    args.pics_dir.mkdir(parents=True, exist_ok=True)

    bbox_images = bbox.load_json(args.bbox_json)

    bbox_images = sorted(bbox_images, key=lambda i: i["path"])
    bbox_images = bbox_images[: args.limit]

    for bbox_image in tqdm(bbox_images):
        with Image.open(bbox_image["path"]) as image:
            draw = ImageDraw.Draw(image)

            for box in bbox_image.boxes:
                x0 = min(box.x0, box.x1)
                x1 = max(box.x0, box.x1)
                y0 = min(box.y0, box.y1)
                y1 = max(box.y0, box.y1)
                draw.rectangle(
                    [(x0, y0), (x1, y1)], outline=bbox.BBOX_COLOR[box["content"]]
                )

            path = args.pics_dir / Path(bbox_image.path).name
            image.save(path)


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
    fix_parser = subparsers.add_parser(
        "fix",
        help="""Edit the bounding box JSON file. Update the directory, rename files,
            update image IDs.""",
    )

    fix_parser.add_argument(
        "--bbox-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Edit this JSON file.""",
    )

    fix_parser.add_argument(
        "--new-dir",
        type=Path,
        metavar="PATH",
        help="""You moved the images to a new directory, possibly to a new computer,
            now update the bug bounding box JSON file to this new directory.""",
    )

    fix_parser.set_defaults(func=fix_action)

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
    pics_parser = subparsers.add_parser(
        "pics",
        help="""Output images with the boxes drawn on them.""",
    )

    pics_parser.add_argument(
        "--bbox-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Use this JSON file as the input.""",
    )

    pics_parser.add_argument(
        "--pics-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the images with boxes to this directory.""",
    )

    pics_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="INT",
        help="""Limit the number of images make. (default: %(default)s)""",
    )

    pics_parser.set_defaults(func=pics_action)

    # ------------------------------------------------------------
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

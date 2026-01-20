#!/usr/bin/env python3

import argparse
import json
import textwrap
from pathlib import Path


def moved_action(args: argparse.Namespace) -> None:
    with args.bbox_json.open() as f:
        images = json.load(f)

    for image in images:
        old = Path(image["path"])
        new = args.new_dir / old.name
        image["path"] = str(new)

    with args.bbox_json.open("w") as f:
        json.dump(images, f, indent=4)


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

    subparsers = argparse.add_subparsers(
        title="Subcommands", description="Actions on bug bounding boxes"
    )

    # ------------------------------------------------------------
    moved_parser = subparsers.add_parser(
        "moved",
        help="Yout moved the images, now update the JSON to use this new directory.",
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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

#!/usr/bin/env python3

import argparse
import textwrap
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import albumentations as alb
import numpy as np
import torch
from datasets import Dataset, Image, NamedSplit
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
from transformers.image_transforms import center_to_corners_format

from moth.pylib import bbox, moth_dataset

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def train_action(args: argparse.Namespace) -> None:
    """
    Train a model.

    Modified from the Hugging Face website:
    https://huggingface.co/docs/transformers/tasks/object_detection
    """
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        do_resize=True,
        size={"max_height": args.image_height, "max_width": args.image_width},
        do_pad=True,
        pad_size={"height": args.image_height, "width": args.image_width},
    )
    train_augment_and_transform = alb.Compose(
        [alb.HorizontalFlip(p=0.5), alb.VerticalFlip(p=0.5)],
        bbox_params=alb.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=25
        ),
    )
    validation_transform = alb.Compose(
        [alb.NoOp()],
        bbox_params=alb.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    train_raw_data = get_dataset("train", args.train_json)
    valid_raw_data = get_dataset("validation", args.valid_json)

    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=image_processor,
    )
    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor,
    )

    train_dataset = train_raw_data.with_transform(train_transform_batch)
    valid_dataset = valid_raw_data.with_transform(validation_transform_batch)

    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name,
        id2label=bbox.id2label,
        label2id=bbox.label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        fp16=False,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
    )

    eval_compute_metrics_fn = partial(
        compute_metrics,
        image_processor=image_processor,
        id2label=bbox.id2label,
        threshold=0.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()


def get_dataset(split: str, dataset_json: Path) -> Dataset:
    records = moth_dataset.MothDataset(bbox_json=dataset_json).to_detr_dict()

    dataset = Dataset.from_dict(
        {
            "image": [r["path"] for r in records],
            "image_id": [r["image_id"] for r in records],
            "width": [r["width"] for r in records],
            "height": [r["height"] for r in records],
            "objects": [r["objects"] for r in records],
        },
        split=NamedSplit(split),
    ).cast_column("image", Image())
    return dataset


def format_image_annotations_as_coco(
    image_id: str,
    categories: list[int],
    areas: list[float],
    bboxes: list[tuple[float]],
) -> dict:
    annotations = []
    for category, area, box in zip(categories, areas, bboxes, strict=True):
        formatted = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(box),
        }
        annotations.append(formatted)
    return {"image_id": image_id, "annotations": annotations}


def convert_bbox_yolo_to_pascal(
    boxes: list[list], image_width: int, image_height: int
) -> list[list]:
    """
    Convert bounding boxes from YOLO to Pascal VOC format.

    YOLO format: (x_center, y_center, width, height) in range [0, 1]
    Pascal VOC format: (x_min, y_min, x_max, y_max) in absolute coordinates
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    boxes = boxes * torch.tensor(
        [[image_width, image_height, image_width, image_height]]
    )

    return boxes


def augment_and_transform_batch(
    examples: dict,
    transform: alb.Compose,
    image_processor: AutoImageProcessor,
    *,
    return_pixel_mask: bool = False,
) -> list[dict]:
    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"], strict=True
    ):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(
        images=images, annotations=annotations, return_tensors="pt"
    )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: list[dict]) -> dict:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics(
    evaluation_results: tuple[list, list],
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: dict[int, str] | None = None,
) -> dict[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    evaluation_results (EvalPrediction): Predictions and targets from evaluation.
    threshold (float, optional): Threshold to filter predicted boxes by confidence.
        Defaults to 0.0.
    id2label (Optional[dict], optional): Mapping from class id to class name.
        Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary
            {<metric_name>: <metric_value>}

    This is lifted straight from the Hugging Face website:

    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dicts with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            height, width = image_target["orig_size"]
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, width, height)
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to
    # Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes, strict=True):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
        )
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(
        classes, map_per_class, mar_100_per_class, strict=True
    ):
        class_name = (
            id2label[class_id.item()] if id2label is not None else class_id.item()
        )
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


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
        "--valid-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""JSON file containing validation data.""",
    )

    train_parser.add_argument(
        "--checkpoint-dir",
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
        "--batch-size",
        type=int,
        default=4,
        metavar="INT",
        help="""How many images to process at a time. (default: %(default)s)""",
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

    train_parser.set_defaults(func=train_action)

    # ------------------------------------------------------------
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.func(ARGS)

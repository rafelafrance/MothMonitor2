from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.transforms import v2

from moth.pylib import bbox


class MothDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        bbox_json: Path | None = None,
        bbox_images: list[bbox.BBoxImage] | None = None,
        transforms: v2.Compose | None = None,
        num_classes: int = bbox.BBOX_NUM_CLASSES,
        limit: int | None = None,
    ) -> None:
        self.num_classes = num_classes

        if bbox_json:
            self.bbox_images = bbox.load_json(bbox_json)
        elif bbox_images:
            self.bbox_images = bbox_images

        if limit:
            self.bbox_images = self.bbox_images[:limit]

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.bbox_images)

    def __getitem__(self, idx: int) -> tuple:
        bbox_image = self.bbox_images[idx]

        image = Image.open(bbox_image.path)

        if bbox_image.bboxes:
            bboxes = tv_tensors.BoundingBoxes(
                bbox_image.bboxes_as_xyxy(), format="XYXY", canvas_size=image.size
            )
        else:
            bboxes = tv_tensors.BoundingBoxes(
                torch.empty((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=image.size,
            )

        labels = torch.as_tensor(
            bbox_image.bbox_labels(self.num_classes), dtype=torch.int64
        )

        if self.transforms is not None:
            image, target = self.transforms(image, bboxes)

        target = {
            "image_id": torch.as_tensor(bbox_image.image_id, dtype=torch.int64),
            "boxes": bboxes,
            "labels": labels,
            "area": torch.as_tensor(bbox_image.bbox_areas(), dtype=torch.float32),
            "iscrowd": torch.zeros((len(bbox_image.bboxes),), dtype=torch.int64),
        }

        return image, target

    def to_detr_dict(self) -> dict:
        records = [
            {
                "image_id": i.image_id,
                "path": i.path,
                "width": i.width,
                "height": i.height,
                "objects": {
                    "id": i.bbox_ids(),
                    "area": i.bbox_areas(),
                    "bbox": i.bboxes_as_xywh(),
                    "category": i.bbox_labels(),
                },
            }
            for i in self.bbox_images
        ]
        return records

    def to_coco_dict(self) -> dict:
        images = []
        annotations = []
        for bbox_image in self.bbox_images:
            images.append(
                {
                    "id": bbox_image.image_id,
                    "file_name": Path(bbox_image.path).name,
                    "height": bbox_image.height,
                    "width": bbox_image.width,
                }
            )
            for box in bbox_image.bboxes:
                annotation = {
                    "id": box.id_,
                    "image_id": bbox_image.image_id,
                    "category_id": bbox.label_to_id(box.content),
                    "bbox": box.xywh(),
                    "area": box.area,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
        coco_dict = {
            "categories": [{"id": i, "name": n} for i, n in bbox.id2label.items()],
            "images": images,
            "annotations": annotations,
        }
        return coco_dict

    def to_coco_obj(self) -> COCO:
        coco_dataset = COCO()
        coco_dataset.dataset = self.to_coco_dict()
        coco_dataset.createIndex()
        return coco_dataset

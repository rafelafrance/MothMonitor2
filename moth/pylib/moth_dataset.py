from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import v2

from moth.pylib import bbox


class MothDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        bbox_json: Path | None = None,
        bbox_images: list[bbox.BBoxImage] | None = None,
        transforms: v2.Compose | None = None,
        limit: int | None = None,
    ) -> None:
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
            bboxes = torch.as_tensor(bbox_image.bboxes_as_xyxy())
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)

        target = {
            "image_id": torch.as_tensor(bbox_image.image_id, dtype=torch.int64),
            "boxes": bboxes,
            "labels": torch.as_tensor(bbox_image.bbox_labels(), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def to_coco_dict(self) -> dict:
        coco_data = {
            "categories": [{"id": i, "name": n} for i, n in bbox.id2label.items()],
            "images": [],
            "annotations": [],
        }
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
                    "category_id": bbox.label2id[box.content],
                    "area": box.area,
                    "bbox": box.xywh(),
                    "iscrowd": 0,
                }
                annotations.append(annotation)
        coco_data["images"] = images
        coco_data["annotations"] = annotations
        return coco_data

    def to_coco_obj(self) -> COCO:
        coco_dataset = COCO()
        coco_dataset.dataset = self.to_coco_dict()
        coco_dataset.createIndex()
        return coco_dataset

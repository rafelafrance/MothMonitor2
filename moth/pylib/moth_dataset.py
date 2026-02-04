from pathlib import Path

import torch
from torchvision import io, tv_tensors

from moth.pylib import bbox


class MothDataset(torch.utils.data.Dataset):
    def __init__(self, bbox_json: Path) -> None:
        self.bbox_images = bbox.load_json(bbox_json)

    def __len__(self) -> int:
        return len(self.bbox_images)

    def __getitem__(self, idx: int) -> tuple:
        bbox_image = self.bbox_images[idx]
        image = io.read_image(bbox_image.path)
        image = tv_tensors.Image(image)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                bbox_image.bboxes_as_xyxy(),
                format="XYXY",
                canvas_size=(bbox_image.height, bbox_image.width),
            ),
            "labels": bbox_image.bbox_labels(),
            "area": bbox_image.bbox_areas(),
            "image_id": idx,
        }

        return image, target

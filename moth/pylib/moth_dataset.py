from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from moth.pylib import bbox


class MothDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bbox_json: Path,
        transforms: v2.Compose | None = None,
        *,
        limit: int | None = None,
    ) -> None:
        self.bbox_images = bbox.load_json(bbox_json)
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
            "boxes": bboxes,
            "labels": torch.as_tensor(bbox_image.bbox_labels(), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

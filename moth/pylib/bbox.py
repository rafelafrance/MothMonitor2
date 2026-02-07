import json
from dataclasses import dataclass, field
from pathlib import Path

TOO_SMALL = 20

FONT: tuple[str, int] = ("liberation sans", 16)
FONT_SM: tuple[str, int] = ("liberation sans", 12)

BBOX_FORMAT: dict[str, dict] = {
    "moth": {"background": "red", "foreground": "white", "font": FONT},
    "not_moth": {"background": "blue", "foreground": "white", "font": FONT},
    "unsure": {"background": "green", "foreground": "white", "font": FONT},
    # {"background": "brown", "foreground": "white", "font": FONT},
    # {"background": "olive", "foreground": "white", "font": FONT},
    # {"background": "teal", "foreground": "white", "font": FONT},
    # {"background": "navy", "foreground": "white", "font": FONT},
    # {"background": "orange", "font": FONT},
    # {"background": "yellow", "font": FONT},
    # {"background": "lime", "font": FONT},
    # {"background": "cyan", "font": FONT},
    # {"background": "purple", "foreground": "white", "font": FONT},
    # {"background": "magenta", "foreground": "white", "font": FONT},
    # {"background": "gray", "font": FONT},
    # {"background": "lavender", "font": FONT},
}
BBOX_COLOR: dict[str, str] = {k: v["background"] for k, v in BBOX_FORMAT.items()}
BBOX_NUM_CLASSES: int = len(BBOX_FORMAT)

id2label: dict[int, str] = dict(enumerate(BBOX_FORMAT))
label2id: dict[str, int] = {k: i for i, k in id2label.items()}


class BBox:
    content: str
    x0: int
    y0: int
    x1: int
    y1: int
    id_: int  # Used by tkinter

    def __init__(
        self, content: str, x0: int, y0: int, x1: int, y1: int, id_: int = 0
    ) -> None:
        self.content = content
        self.x0 = min(x0, x1)
        self.y0 = min(y0, y1)
        self.x1 = max(x0, x1)
        self.y1 = max(y0, y1)
        self.id_ = id_

    @classmethod
    def load_json(cls, bbox_data: dict) -> "BBox":
        b = bbox_data
        return cls(
            content=b["content"],
            x0=b["x0"],
            y0=b["y0"],
            x1=b["x1"],
            y1=b["y1"],
            id_=b["id_"],
        )

    @property
    def label(self) -> int:
        return label2id[self.content]

    @property
    def area(self) -> float:
        return float((self.x1 - self.x0) * (self.y1 - self.y0))

    def too_small(self) -> bool:
        return (self.x1 - self.x0) < TOO_SMALL or (self.y1 - self.y0) < TOO_SMALL

    def point_hit(self, x: int, y: int) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def xyxy(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def xywh(self) -> list[int]:
        return [self.x0, self.y0, (self.x1 - self.x0), (self.y1 - self.y0)]

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BBoxImage:
    path: str
    width: int = 0  # It's just easier to have this here
    height: int = 0  # It's just easier to have this here
    image_id: int = 0
    bboxes: list[BBox] = field(default_factory=list)

    def filter_size(self) -> None:
        self.bboxes = [b for b in self.bboxes if not b.too_small()]

    def delete_box(self, x: int, y: int) -> None:
        if hits := [b for b in self.bboxes if b.point_hit(x, y)]:
            hits = sorted(hits, key=lambda b: b.area)
            self.bboxes = [b for b in self.bboxes if b != hits[0]]

    @classmethod
    def load_json(cls, bbox_data: dict) -> "BBoxImage":
        return cls(
            path=bbox_data["path"],
            width=bbox_data["width"],
            height=bbox_data["height"],
            image_id=bbox_data.get("image_id", 0),
            bboxes=[BBox.load_json(b) for b in bbox_data["bboxes"]],
        )

    def bboxes_as_xyxy(self) -> list[list[int]]:
        return [b.xyxy() for b in self.bboxes]

    def bboxes_as_xywh(self) -> list[list[int]]:
        return [b.xywh() for b in self.bboxes]

    def bbox_labels(self) -> list[int]:
        return [b.label for b in self.bboxes]

    def bbox_areas(self) -> list[float]:
        return [b.area for b in self.bboxes]

    def to_dict(self) -> dict:
        dct = {k: v for k, v in self.__dict__.items() if k != "bboxes"}
        dct["bboxes"] = [b.to_dict() for b in self.bboxes]
        return dct


def load_json(bbox_json: Path) -> list[BBoxImage]:
    with bbox_json.open() as f:
        bbox_data = json.load(f)
    return [BBoxImage.load_json(d) for d in bbox_data]


def dump_json(bbox_images: list[BBoxImage], bbox_json: Path, indent: int = 4) -> None:
    json_data = [i.to_dict() for i in bbox_images]
    with bbox_json.open("w") as f:
        json.dump(json_data, f, indent=indent)

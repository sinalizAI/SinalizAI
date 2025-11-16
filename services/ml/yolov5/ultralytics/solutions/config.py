

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2


@dataclass
class SolutionConfig:
    

    source: Optional[str] = None
    model: Optional[str] = None
    classes: Optional[List[int]] = None
    show_conf: bool = True
    show_labels: bool = True
    region: Optional[List[Tuple[int, int]]] = None
    colormap: Optional[int] = cv2.COLORMAP_DEEPGREEN
    show_in: bool = True
    show_out: bool = True
    up_angle: float = 145.0
    down_angle: int = 90
    kpts: List[int] = field(default_factory=lambda: [6, 8, 10])
    analytics_type: str = "line"
    figsize: Optional[Tuple[int, int]] = (12.8, 7.2)
    blur_ratio: float = 0.5
    vision_point: Tuple[int, int] = (20, 20)
    crop_dir: str = "cropped-detections"
    json_file: str = None
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: Optional[str] = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs):
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f" {key} is not a valid solution argument, available arguments here: https://docs.ultralytics.com/solutions/#solutions-arguments"
                )
        return self

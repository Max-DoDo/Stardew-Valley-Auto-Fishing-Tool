
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from src.log import Log

Monitor = dict[str, int]
BBox = tuple[int, int, int, int]
Point = tuple[int, int]

@dataclass
class FishDetectionResult:
    found: bool
    template_name: Optional[str] = None
    confidence: float = 0.0
    scale: float = 0.0

    fish_bbox: Optional[BBox] = None
    fish_center: Optional[Point] = None
    roi_bbox: Optional[BBox] = None
    roi_monitor: Optional[Monitor] = None
    
class FishDetector:

    def __init__(self,template_dir: str | Path = "assets/fish_templates",threshold: float = 0.70,):
        self.scales = (
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,)
        self.fish_template_dir = Path(template_dir)
        self.threshold = threshold
        self.fish_templates: list[tuple[str, np.ndarray]] = []
        self._load_fish_templates()

    def _load_fish_templates(self):
        self.fish_templates.clear()

        if not self.fish_template_dir.exists():
            Log.warn(f"找不到 Fish 模板文件夹: {self.fish_template_dir}")
            raise FileNotFoundError(
            )
    
        image_paths = []
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(self.fish_template_dir.glob(suffix))

        if not image_paths:
            Log.error(f"Fish模板文件夹中没有图片: {self.fish_template_dir}")
            raise FileNotFoundError(
            )

        for image_path in image_paths:
            template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                Log.warn(f"无法读取 Fish 模板: {image_path}")
                continue
            self.fish_templates.append((image_path.stem, template))
            Log.info(f"加载 Fish 模板: {image_path}")

        if not self.fish_templates:
            Log.error("没有成功加载任何鱼图标模板。")
            raise RuntimeError(
            )
        Log.success(f"已加载 {len(self.fish_templates)} 个 Fish 模板")

    def detect(self, frame: np.array, monitor: Monitor):
        if frame is None or frame.size == 0:
            return FishDetectionResult(found=False)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_confidence = 0.0
        best_template_name: Optional[str] = None
        best_scale = 1.0
        best_fish_bbox: Optional[BBox] = None

        for template_name, template in self.fish_templates:
            match_result = self._match_template_multiscale(gray_frame=gray_frame, template=template)

            if match_result is None:
                continue

            fish_bbox, confidence, scale = match_result

            if confidence > best_confidence:
                best_template_name = template_name
                best_confidence = confidence
                best_scale = scale
                best_fish_bbox = fish_bbox

        if best_confidence < self.threshold:
            return FishDetectionResult(
                found=False,
                template_name=best_template_name,
                confidence=best_confidence,
                scale=best_scale,
                fish_bbox=best_fish_bbox,
            )

        return FishDetectionResult(
            found=True,
            template_name=best_template_name,
            confidence=best_confidence,
            scale=best_scale,
            fish_bbox=best_fish_bbox,
            fish_center = ( 
                best_fish_bbox[0] + best_fish_bbox[2] // 2,
                best_fish_bbox[1] + best_fish_bbox[3] // 2
            ),
            roi_monitor=self._bbox_to_screen_monitor(fish_bbox, monitor)
        )

    def _bbox_to_screen_monitor(
        self,
        bbox: BBox,
        monitor: Monitor,
    ) -> Monitor:
            
        screen_origin = (
            monitor["left"],
            monitor["top"],
        )

        origin_x, origin_y = screen_origin

        x, y, w, h = bbox

        return {
            "left": int(origin_x + x),
            "top": int(origin_y + y),
            "width": int(w),
            "height": int(h),
        }
    
    def _match_template_multiscale(
        self,
        gray_frame: np.ndarray,
        template: np.ndarray,
    ) -> Optional[tuple[BBox, float, float]]:
        
        frame_h, frame_w = gray_frame.shape[:2]
        template_h, template_w = template.shape[:2]

        best_confidence = 0.0
        best_bbox: Optional[BBox] = None
        best_scale = 1.0

        for scale in self.scales:
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)

            if scaled_w <= 0 or scaled_h <= 0:
                continue

            if scaled_w > frame_w or scaled_h > frame_h:
                continue

            interpolation = (
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_LINEAR
            )

            scaled_template = cv2.resize(
                template,
                (scaled_w, scaled_h),
                interpolation=interpolation,
            )

            result = cv2.matchTemplate(
                gray_frame,
                scaled_template,
                cv2.TM_CCOEFF_NORMED,
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            confidence = float(max_val)

            if confidence > best_confidence:
                x, y = max_loc

                best_confidence = confidence
                best_bbox = (
                    int(x),
                    int(y),
                    int(scaled_w),
                    int(scaled_h),
                )
                best_scale = scale

        if best_bbox is None:
            return None

        return best_bbox, best_confidence, best_scale

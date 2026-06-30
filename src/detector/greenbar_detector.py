from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import cv2
import mss
import numpy as np
import pygetwindow as gw

from src.tools.config_manager import ConfigManager
from src.tools.log import Log

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class GreenBarResult:
    found: Optional[bool] = None
    greenbar_bbox: Optional[BBox] = None
    greenbar_center: Optional[Point] = None
    confidence:  Optional[float] = None

class GreenBarDetector:

    def __init__(
        self,
        config_path: str | Path = "config.json",

    ):
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.load()

        greenbar = self.config["green_bar"]
        self.greenbar_hsv_high = np.array(greenbar["hsv_high"], dtype=np.uint8)
        self.greenbar_hsv_low = np.array(greenbar["hsv_low"], dtype=np.uint8)

        self.bar_size_locked = False
        self.bar_size_samples: list[BBox] = []
        self.bar_size_sample_limit = 20

        self.last_bar_center: Point | None = None

        self.lastResult = None
        self.missing_bar_frames = 0
        self.max_missing_bar_frames = 20  

    def get_green_bar_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.greenbar_hsv_low,
            self.greenbar_hsv_high,
        )

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 竖向闭运算：连接上下断开的白色区域
        vertical_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 30)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            vertical_close_kernel
        )

        return mask

    def detect(self, frame : np.ndarray):
        mask = self.get_green_bar_mask(frame)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[tuple[float, BBox]] = []

        # Log.debug("=====Frame=====")
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            min_fill_ratio = 0.40
            rect_area = bw * bh
            # 寻找竖向矩形
            aspect_ratio = bh / max(bw, 1)

            if aspect_ratio < 1.2:
                continue

            # 1. 面积太小
            if area < 100:
                continue

            # 5. 填充率：bbox 里面真正绿色像素占多少
            roi_mask = mask[y:y + bh, x:x + bw]
            green_pixels = cv2.countNonZero(roi_mask)

            fill_ratio = green_pixels / rect_area

            if fill_ratio < min_fill_ratio:
                continue

            score = area * aspect_ratio
            if self.last_bar_center is not None:
                last_cx, last_cy = self.last_bar_center
                cx = x + bw // 2
                cy = y + bh // 2
                y_weight = 0.3
                distance = np.hypot(cx - last_cx, (cy - last_cy) * y_weight)
                score -= distance * 200
                # Log.debug(f"GBDetector: {x, y, bw, bh} score: {score}")

            candidates.append((score, (x, y, bw, bh)))

        if not candidates:
            self.missing_bar_frames += 1
            if self.missing_bar_frames >= self.max_missing_bar_frames:
                if self.bar_size_locked:
                    self.reset_bar_tracking()
                    self.missing_bar_frames = 0
            elif self.lastResult is not None:
                # Log.debug("重建钓鱼条")
                return self.lastResult
            return GreenBarResult(found=False)


        
        self.missing_bar_frames = 0
        # 选择最像钓鱼条的轮廓
        candidates.sort(key=lambda item: item[0], reverse=True)

        best_score, bbox = candidates[0]
        x, y, bw, bh = bbox

        center = (
            x + bw // 2,
            y + bh // 2,
        )
        raw_bbox = bbox

        self._update_bbox_size_lock(raw_bbox)

        if self.bar_size_locked:
            final_bbox = self._make_locked_bar_bbox(center)
        else:
            final_bbox = raw_bbox

        confidence = min(1.0, best_score / 5000)
        self.last_bar_center = center
        self.lastResult = GreenBarResult(
            found=True,
            greenbar_center=center,
            greenbar_bbox=final_bbox,
            confidence=confidence
        )
        return self.lastResult

    def _update_bbox_size_lock(self, bbox: BBox):
        if self.bar_size_locked:
            return
        
        self.bar_size_samples.append(bbox)

        if len(self.bar_size_samples) < self.bar_size_sample_limit:
            return
        
        widths = np.array([box[2] for box in self.bar_size_samples], dtype=np.float32)
        heights = np.array([box[3] for box in self.bar_size_samples], dtype=np.float32)

        locked_w = int(np.median(widths))
        locked_h = int(np.max(heights))
        
        self.locked_bar_width = max(1, locked_w)
        self.locked_bar_height = max(1, locked_h)
        self.bar_size_locked = True

        Log.success(
        f"已锁定钓鱼条大小: "
        f"width={self.locked_bar_width}, "
        f"height={self.locked_bar_height}"
        )

    def _make_locked_bar_bbox(self, center: Point) -> BBox:
        """
        使用锁定后的钓鱼条大小，根据当前中心点重建 bbox。
        """
        if self.locked_bar_width is None or self.locked_bar_height is None:
            raise RuntimeError("钓鱼条大小尚未锁定")

        cx, cy = center

        x = int(cx - self.locked_bar_width / 2)
        y = int(cy - self.locked_bar_height / 2)

        return (
            x,
            y,
            self.locked_bar_width,
            self.locked_bar_height,
        )
    
    def reset_bar_tracking(self) -> None:
        """
        重置钓鱼条跟踪状态。
        新一轮钓鱼开始时调用。
        """
        self.bar_size_locked = False
        self.locked_bar_width = None
        self.locked_bar_height = None
        self.bar_size_samples.clear()
        self.last_bar_center = None
        self.lastResult = None

        Log.info("已重置钓鱼条跟踪状态")

        
        
            

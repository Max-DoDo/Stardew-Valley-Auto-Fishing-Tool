from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import cv2
import mss
import numpy as np
import pygetwindow as gw

from src.config_manager import ConfigManager
from src.log import Log

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class GreenBarResult:
    found: Optional[bool] = None
    greenbar_bbox: Optional[BBox] = None
    greenbar_center: Optional[Point] = None
    confidence:  Optional[float] = None
    # found_fish: Optional[bool] = None
    # found_pbox: Optional[bool] = None


    # fish_location: Optional[Point] = None
    # fish_confidence: Optional[float] = None

    # pbox_confidence:  Optional[float] = None
    # template_name: Optional[str] = None


class GreenBarDetector:

    def __init__(
        self,
        config_path: str | Path = "config.json",
        fish_template_dir: str | Path = "assets/fish_templates",
    ):
        self.sct = mss.MSS()
        
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

        self.missing_bar_frames = 0
        self.max_missing_bar_frames = 40  

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
            (1, 29)
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

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            min_fill_ratio = 0.70
            rect_area = bw * bh
            # 寻找竖向矩形
            aspect_ratio = bh / max(bw, 1)

            if aspect_ratio < 1.3:
                continue

            # 1. 面积太小，直接排除
            if area < 500:
                continue

            # 5. 填充率：bbox 里面真正绿色像素占多少
            roi_mask = mask[y:y + bh, x:x + bw]
            green_pixels = cv2.countNonZero(roi_mask)

            fill_ratio = green_pixels / rect_area

            if fill_ratio < min_fill_ratio:
                continue

            score = area * aspect_ratio
            candidates.append((score, (x, y, bw, bh)))

        if not candidates:
            self.missing_bar_frames += 1
            if self.missing_bar_frames >= self.max_missing_bar_frames:
                if self.bar_size_locked:
                    self.reset_bar_tracking()
                    self.missing_bar_frames = 0
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

        return GreenBarResult(
            found=True,
            greenbar_center=center,
            greenbar_bbox=final_bbox,
            confidence=confidence
        )

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

        Log.info("已重置钓鱼条跟踪状态")
    
    def preview(self) -> None:
        """
        预览 ROI 截图，并显示截图 FPS 和实际 FPS。
        """
        import time

        actual_fps = 0.0
        capture_fps = 0.0
        last_frame_time = time.perf_counter()

        Log.info(f"开始预览 ROI, 目标 FPS = {self.target_fps}")

        preview_win = "Fishing Bar Detection Preview"
        mask_win = "Green Bar Mask"

        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(mask_win, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(preview_win, self.roi_width, self.roi_height)
        cv2.resizeWindow(mask_win, self.roi_width, self.roi_height)
        # 置顶窗口
        try:
            cv2.setWindowProperty(preview_win, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(mask_win, cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            Log.warn(f"设置窗口置顶失败: {e}")

        while True:
            loop_start = time.perf_counter()

            capture_start = time.perf_counter()
            frame = self.capture_roi()
            capture_elapsed = time.perf_counter() - capture_start

            if capture_elapsed > 0:
                capture_fps = 1.0 / capture_elapsed

            # 检测绿色钓鱼条
            greenbar_result = self.detect(frame)

            # 生成 mask，用于调 HSV
            mask = self.get_green_bar_mask(frame)

            debug_frame = frame.copy()

            if (
                greenbar_result.found
                and greenbar_result.greenbar_bbox is not None
                and greenbar_result.greenbar_center is not None
            ):
                x, y, bw, bh = greenbar_result.greenbar_bbox
                cx, cy = greenbar_result.greenbar_center

                cv2.rectangle(
                    debug_frame,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    debug_frame,
                    (cx, cy),
                    4,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    debug_frame,
                    f"BAR FOUND y={cy} conf={greenbar_result.confidence:.2f}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    debug_frame,
                    "BAR NOT FOUND",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                debug_frame,
                f"Actual FPS: {actual_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_frame,
                f"Capture FPS: {capture_fps:.1f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(preview_win, debug_frame)
            cv2.imshow(mask_win, mask)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = self.frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            now = time.perf_counter()
            frame_time = now - last_frame_time
            last_frame_time = now

            if frame_time > 0:
                current_actual_fps = 1.0 / frame_time
                actual_fps = actual_fps * 0.5 + current_actual_fps * 0.5

        cv2.destroyAllWindows()


if __name__ == "__main__":
    fv = GreenBarDetector()
    fv.preview()

    # fv.detect_fishing_bar(fv.capture_roi())

        
        
            

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
class BboxResult:
    found_bbox: Optional[bool] = None
    bbox_location: Optional[BBox] = None
    bbox_center: Optional[Point] = None
    bbox_confidence:  Optional[float] = None
    # found_fish: Optional[bool] = None
    # found_pbox: Optional[bool] = None


    # fish_location: Optional[Point] = None
    # fish_confidence: Optional[float] = None

    # pbox_confidence:  Optional[float] = None
    # template_name: Optional[str] = None


class BboxDetector:

    def __init__(
        self,
        config_path: str | Path = "config.json",
        fish_template_dir: str | Path = "assets/fish_templates",
    ):
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(self.config_path)
        self.sct = mss.MSS()

        self.config = self.config_manager.load()

        self.window_title: str = self.config["window_title"]

        roi = self.config["roi"]
        self.roi_left: int = roi["left"]
        self.roi_top: int = roi["top"]
        self.roi_width: int = roi["width"]
        self.roi_height: int = roi["height"]
        self.target_fps = self.config["target_fps"]
        self.frame_interval = 1.0 / self.target_fps

        greenbar = self.config["green_bar"]
        self.greenbar_hsv_high = np.array(greenbar["hsv_high"], dtype=np.uint8)
        self.greenbar_hsv_low = np.array(greenbar["hsv_low"], dtype=np.uint8)

        self.fish_template_dir = Path(fish_template_dir)
        self.fish_templates: list[tuple[str, np.ndarray]] = []
        self._load_fish_templates()
        self.game_window = self.find_window()
        self.monitor = {
            "left": self.game_window.left + self.roi_left,
            "top": self.game_window.top + self.roi_top,
            "width": self.roi_width,
            "height": self.roi_height,
        }
        self.bar_size_locked = False
        self.bar_size_samples: list[BBox] = []
        self.bar_size_sample_limit = 20

        self.last_bar_center: Point | None = None

        self.missing_bar_frames = 0
        self.max_missing_bar_frames = 40

        Log.info(f"当前monitor大小 {self.monitor}")

    def _load_fish_templates(self) -> None:
        
        if not self.fish_template_dir.exists():
            Log.error(f"找不到鱼图标文件夹{self.fish_template_dir}")
            raise FileNotFoundError(
            )

        image_paths = []

        for suffix in("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(self.fish_template_dir.glob(suffix))

        if not image_paths:
            Log.error(f"模板文件夹中没有图片: {self.fish_template_dir}")
            raise FileNotFoundError(
            )
        
        for image_path in image_paths:
            temp = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if temp is None:
                Log.warn(f"无法读取模板{image_path}")
                continue
            Log.info(f"加载鱼图标模板{image_path}")
            self.fish_templates.append((image_path.stem, temp))

        if not self.fish_templates:
            Log.error("没有成功加载任何鱼图标模板。")
            raise RuntimeError(
            )
        
        Log.success(f"已加载 {len(self.fish_templates)} 个鱼图标模板。")


    def find_window(self):
        all_windows = gw.getAllWindows()

        for window in all_windows:
            title = window.title or ""
            Log.info(f"正在寻找游戏窗口{title}")

            if self.window_title.lower() == title.lower():
                Log.success(f"匹配游戏窗口 标题:{title}")
                return window
            
        Log.warn(f"Could not find game window")
        raise RuntimeError()

    def capture_roi(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame
    

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

    def detect_fishing_bar(self, frame : np.ndarray):
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

            # 寻找竖向矩形
            aspect_ratio = bh / max(bw, 1)

            if aspect_ratio < 1.2:
                continue
            score = area * aspect_ratio
            candidates.append((score, (x, y, bw, bh)))

        if not candidates:
            self.missing_bar_frames += 1
            if self.missing_bar_frames >= self.max_missing_bar_frames:
                if self.bar_size_locked:
                    self.reset_bar_tracking()
                    self.missing_bar_frames = 0
            return BboxResult(found_bbox=False)
        
        self.missing_bar_frames = 0
        # 选择最像钓鱼条的轮廓
        candidates.sort(key=lambda item: item[0], reverse=True)

        best_score, bbox = candidates[0]
        x, y, bw, bh = bbox

        center = (
            x + bw // 2,
            y + bh // 2,
        )

        # 当前帧原始检测结果
        raw_bbox = bbox
        # raw_center = center

        # 用当前 bbox 更新大小锁定器
        self._update_bbox_size_lock(raw_bbox)

        # # 平滑 center，减少鱼图标造成的中心抖动
        # if self.last_bar_center is not None:
        #     old_x, old_y = self.last_bar_center
        #     new_x, new_y = raw_center

        #     alpha = 0.35

        #     smooth_center = (
        #         int(old_x * (1 - alpha) + new_x * alpha),
        #         int(old_y * (1 - alpha) + new_y * alpha),
        #     )
        # else:
        #     smooth_center = raw_center

        # self.last_bar_center = smooth_center

        # 如果大小已经锁定，就不要再用当前帧 bbox 的宽高
        if self.bar_size_locked:
            final_bbox = self._make_locked_bar_bbox(center)
        else:
            final_bbox = raw_bbox

        confidence = min(1.0, best_score / 5000)

        return BboxResult(
            found_bbox=True,
            bbox_center=center,
            bbox_location=final_bbox,
            bbox_confidence=confidence
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
            greenbar_result = self.detect_fishing_bar(frame)

            # 生成 mask，用于调 HSV
            mask = self.get_green_bar_mask(frame)

            debug_frame = frame.copy()

            if (
                greenbar_result.found_bbox
                and greenbar_result.bbox_location is not None
                and greenbar_result.bbox_center is not None
            ):
                x, y, bw, bh = greenbar_result.bbox_location
                cx, cy = greenbar_result.bbox_center

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
                    f"BAR FOUND y={cy} conf={greenbar_result.bbox_confidence:.2f}",
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
    fv = BboxDetector()
    fv.preview()

    # fv.detect_fishing_bar(fv.capture_roi())

        
        
            

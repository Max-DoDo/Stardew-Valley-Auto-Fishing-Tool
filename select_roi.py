import ctypes
import time
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import mss
import numpy as np
import pygetwindow as gw
from config_manager import ConfigManager
from log import Log


ROI = Tuple[int, int, int, int]

class WindowNotFoundError(Exception):
    """
    没有找到游戏窗口时抛出的异常。
    """
    pass

class FishingROISelector:
    """
    用于：
    1. 查找星露谷物语窗口
    2. 截图游戏窗口
    3. 手动框选钓鱼小游戏 ROI
    4. 自动写入 config.json
    """

    def __init__(
        self,
        window_keywords: Tuple[str, ...] = ("Stardew Valley", "星露谷物语"),
        config_path: Path | str = "config.json",
        capture_output: str = "game_window.png",
        max_display_w: int = 1200,
        max_display_h: int = 800,
        activate_delay=0.5,
    ):
        self.window_keywords = window_keywords
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(self.config_path)
        self.capture_output = capture_output

        self.max_display_w = max_display_w
        self.max_display_h = max_display_h
        self.activate_delay = activate_delay

        self.window: Optional[Any] = None
        self.frame: Optional[np.ndarray] = None

        self.scale: float = 1.0
        self.display: Optional[np.ndarray] = None
        self.display_w: int = 0
        self.display_h: int = 0

        self.dragging: bool = False
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.roi: Optional[ROI] = None

        self.window_name = "Select Fishing ROI"

    @staticmethod
    def set_dpi_awareness() -> None:
        """
        避免 Windows 高 DPI 缩放导致坐标不准。
        """
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    def find_window(self, keywords):
        """
        根据窗口标题寻找窗口。
        """
        all_windows = gw.getAllWindows()

        matched = []

        for window in all_windows:
            title = window.title or ""

            for keyword in keywords:
                if keyword.lower() in title.lower():
                    matched.append(window)
                    break

        if not matched:
            visible_titles = [
            window.title
            for window in all_windows
            if window.title
        ]
            
            Log.warn(f"没有找到包含 {keywords} 的窗口。请确保该窗口不是独占全屏模式")
            Log.warn("当前可见窗口标题: ")
            Log.warn("\n".join(f"- {title}" for title in visible_titles))
            raise WindowNotFoundError("No available window")

        matched.sort(key=lambda w: w.width * w.height, reverse=True)
        return matched[0]

    def bring_window_to_front(self,window) -> None:
        """
        把指定窗口拉到前台。
        """
        try:
            if window.isMinimized:
                window.restore()

            window.activate()
            time.sleep(self.activate_delay)

        except Exception as e:
            Log.warn("尝试拉起窗口时出现问题", e)

    def capture_window(self, window) -> np.ndarray:
        """
        截取整个游戏窗口，包含窗口边框。
        ROI 后续会以窗口左上角为基准保存。
        """
        monitor = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }

        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cv2.imwrite(self.capture_output, frame)
        return frame

    @staticmethod
    def _normalize_rect(x1: int, y1: int, x2: int, y2: int) -> ROI:
        """
        把任意拖拽方向的矩形转成标准 x, y, w, h。
        """
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        return x, y, w, h

    def prepare_display_image(self) -> None:
        """
        准备用于 ROI 选择的显示图像。
        如果截图太大，则缩小显示，但保存真实坐标。
        """
        if self.frame is None:
            raise RuntimeError("frame 为空，请先调用 capture_window。")

        original_h, original_w = self.frame.shape[:2]

        self.scale = min(
            1.0,
            self.max_display_w / original_w,
            self.max_display_h / original_h,
        )

        self.display_w = int(original_w * self.scale)
        self.display_h = int(original_h * self.scale)

        self.display = cv2.resize(self.frame, (self.display_w, self.display_h))

    def display_to_real(self, x: int, y: int) -> Tuple[int, int]:
        """
        把显示窗口坐标转换成真实截图坐标。
        """
        return int(x / self.scale), int(y / self.scale)

    def reset_selection(self) -> None:
        """
        重置当前 ROI 选择状态。
        """
        self.dragging = False
        self.start = None
        self.end = None
        self.roi = None

    def mouse_callback(self, event, x, y, flags, param) -> None:
        """
        OpenCV 鼠标回调函数。
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
            self.roi = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end = (x, y)

            if self.start is None or self.end is None:
                return

            sx, sy = self.start
            ex, ey = self.end

            x1, y1 = self.display_to_real(sx, sy)
            x2, y2 = self.display_to_real(ex, ey)

            rx, ry, rw, rh = self._normalize_rect(x1, y1, x2, y2)

            if rw > 5 and rh > 5:
                self.roi = (rx, ry, rw, rh)

    def get_current_roi(self) -> Optional[ROI]:
        """
        获取当前正在拖拽或已经确认的 ROI。
        """
        if self.start is None or self.end is None:
            return self.roi

        sx, sy = self.start
        ex, ey = self.end

        dx, dy, dw, dh = self._normalize_rect(sx, sy, ex, ey)

        real_x1, real_y1 = self.display_to_real(dx, dy)
        real_x2, real_y2 = self.display_to_real(dx + dw, dy + dh)

        current_roi = self._normalize_rect(real_x1, real_y1, real_x2, real_y2)

        if self.roi:
            current_roi = self.roi

        return current_roi

    def draw_roi_rectangle(self, canvas: np.ndarray) -> None:
        """
        在显示图像上画出当前 ROI 框。
        """
        if self.start is None or self.end is None:
            return

        sx, sy = self.start
        ex, ey = self.end

        dx, dy, dw, dh = self._normalize_rect(sx, sy, ex, ey)

        cv2.rectangle(
            canvas,
            (dx, dy),
            (dx + dw, dy + dh),
            (0, 255, 0),
            2,
        )

    def draw_info_panel(self, canvas: np.ndarray, current_roi: Optional[ROI]) -> None:
        """
        绘制左上角信息面板。
        """
        if self.window is None:
            raise RuntimeError("window 为空。")

        panel_h = 100
        cv2.rectangle(canvas, (0, 0), (self.display_w, panel_h), (0, 0, 0), -1)

        if current_roi:
            rx, ry, rw, rh = current_roi

            screen_left = self.window.left + rx
            screen_top = self.window.top + ry

            lines = [
                f"Window-relative ROI: x={rx}, y={ry}, w={rw}, h={rh}",
                f"Screen position: left={screen_left}, top={screen_top}",
                "Enter/Space = save | R = reset | Esc/Q = quit",
            ]
        else:
            lines = [
                "Drag with left mouse button to select the fishing mini-game area.",
                "Enter/Space = save | R = reset | Esc/Q = quit",
            ]

        y = 25

        for line in lines:
            cv2.putText(
                canvas,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 32

    def select_roi(self) -> ROI:
        """
        自定义 ROI 选择器：
        - 鼠标左键拖拽选择区域
        - 实时显示坐标和尺寸
        - Enter / Space 保存
        - R 重选
        - Esc / Q 退出
        """
        self.prepare_display_image()

        if self.display is None:
            raise RuntimeError("display 为空。")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n操作说明：")
        print("鼠标左键拖拽：框选钓鱼小游戏区域")
        print("Enter / Space：保存到 config.json")
        print("R：重新选择")
        print("Esc / Q：退出，不保存\n")

        while True:
            canvas = self.display.copy()

            current_roi = self.get_current_roi()

            self.draw_roi_rectangle(canvas)
            self.draw_info_panel(canvas, current_roi)

            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):  # Enter 或 Space
                if not current_roi:
                    Log.info("你还没有框选区域。")
                    continue

                return current_roi

            elif key in (ord("r"), ord("R")):
                self.reset_selection()
                Log.info("已重置，请重新框选。")

            elif key in (27, ord("q"), ord("Q")):
                Log.info("已退出，没有保存。")
                raise SystemExit

    def minimize_game_window(self) -> None:
        """
        截图完成后，把游戏窗口最小化，避免挡住 ROI 选择窗口。
        """
        if self.window is None:
            return

        try:
            self.window.minimize()
            time.sleep(0.3)
        except Exception as e:
            Log.warn("尝试最小化游戏窗口失败，但可以继续：", e)

    def run(self) -> ROI:
        """
        执行完整流程。
        """
        self.set_dpi_awareness()

        Log.info("正在寻找星露谷物语窗口...")
        try:
            self.window = self.find_window(self.window_keywords)
        except Exception as e:
            Log.error(e)
            return -1,-1,-1,-1
        self.bring_window_to_front(self.window)

        Log.success(f"找到窗口：{self.window.title}")
        Log.info(
            f"窗口位置: left={self.window.left}, "
            f"top={self.window.top}, "
            f"width={self.window.width}, "
            f"height={self.window.height}"
        )


        Log.info("正在截图游戏窗口...")
        self.frame = self.capture_window(self.window)

        Log.info(f"窗口截图已保存为：{self.capture_output}")

        self.minimize_game_window()

        try:
            roi_left, roi_top, roi_width, roi_height = self.select_roi()
            # self.bring_window_to_front(self.find_window(self.window_name))
        finally:
            cv2.destroyAllWindows()

        values = {
            "window_title": self.window.title,
            "roi": {
                "left": roi_left,
                "top": roi_top,
                "width": roi_width,
                "height": roi_height,
            }
        }

        self.config_manager.update(values)

        Log.info(f"已写入 {self.config_path}")
        Log.info(f"window_title = {self.window.title!r}")
        Log.info(f"roi.left = {roi_left}")
        Log.info(f"roi.top = {roi_top}")
        Log.info(f"roi.width = {roi_width}")
        Log.info(f"roi.height = {roi_height}")

        return roi_left, roi_top, roi_width, roi_height


if __name__ == "__main__":
    selector = FishingROISelector()
    selector.run()
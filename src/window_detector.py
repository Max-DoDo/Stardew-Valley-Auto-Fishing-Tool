import time
from pathlib import Path
from typing import Optional
import cv2
import mss
import numpy as np
import pygetwindow as gw
from src.config_manager import ConfigManager
from src.log import Log
import win32gui # type: ignore
import win32con # type: ignore

Monitor = dict[str, int]

class WindowDetector:
        
    def __init__(self,
                config_path: str | Path = "config.json",
    ):
        self.sct = mss.MSS()

        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.load()

        self.window_title = self.config["window_title"]
        self.window = None
    
    def find_window(self):
        all_windows = gw.getAllWindows()

        for window in all_windows:
            title = window.title or ""
            # Log.info(f"寻找窗口: {title}")

            if self.window_title.lower() == title.lower():
                self.window = window
                Log.success(f"匹配到窗口,标题: {title}")
                return window
            
        Log.warn(f"Could not find game window")
        return None
    
    def get_hwnd(self) -> int:
        if self.window is None:
            self.find_window()

        hwnd = getattr(self.window, "_hWnd", None)

        if hwnd is None:
            raise RuntimeError("无法从 pygetwindow 获取 hwnd")

        return hwnd
    
    def restore_if_minimized(self) -> None:
        hwnd = self.get_hwnd()

        if win32gui.IsIconic(hwnd):
            Log.warn("窗口已最小化，正在恢复")
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.3)
    
    def capture_roi(self, monitor: dict):
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def capture(self):
        screenshot = self.sct.grab(self.get_monitor())
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def get_monitor(self) -> Monitor:
        if self.window is None:
            self.find_window()

        self.restore_if_minimized()

        hwnd = self.get_hwnd()

        left, top, right, bottom = win32gui.GetClientRect(hwnd)

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"客户区大小异常: width={width}, height={height}"
            )

        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))

        monitor: Monitor = {
            "left": int(screen_left),
            "top": int(screen_top),
            "width": int(width),
            "height": int(height),
        }
        # Log.info(f"正在获取窗口信息: {monitor}")
        self.monitor = monitor
        return monitor
    
    

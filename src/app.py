from src.log import Log
from src.ui_detector import UIDetector
import pygetwindow as gw

class App:

    def __init__():
        pass

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

if __name__ == "__main__":
    ud =UIDetector()
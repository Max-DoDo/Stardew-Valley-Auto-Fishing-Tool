from pathlib import Path

import cv2
import numpy as np
from fish_detector import FishDetector
from src.greenbar_detector import GreenBarDetector
from src.config_manager import ConfigManager
from src.log import Log
from src.ui_detector import UIDetector
from src.window_detector import WindowDetector
import time


class App:

    def __init__(self,
                config_path: str | Path = "config.json"
                ):
        
        # Load config
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.load()

        # fps
        self.target_fps = self.config["target_fps"]
        self.frame_interval = 1.0 / self.target_fps

        self.last_preview_size = None

        self.findUIROI = False
        self.is_update = True
        self.monitor = None
        self.roi = None

        self.wd = WindowDetector()
        self.ud = UIDetector()
        self.gd = GreenBarDetector()#
        self.fd = FishDetector()

        window = self.wd.find_window()

        while True:
            if window is None:
                sleeptime = 3
                Log.warn(f"Retry in {sleeptime} second")
                time.sleep(sleeptime)
                window = self.wd.find_window()
            else:
                break

    def run(self):

        Log.info(f"开始预览 ROI, 目标 FPS = {self.target_fps}")

        preview_win = "Detection Preview"
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(preview_win, cv2.WND_PROP_TOPMOST, 1)


        while True:
            # startTime = time.perf_counter()
            self.update_monitor()
            if self.is_update or not self.findUIROI:
                self.update_ui()
            
            if self.roi is None:
                roi = self.monitor
                self._resize_preview_window(preview_win,500, 300)
            else:
                roi = self.roi
                self._resize_preview_window(preview_win,roi["width"], roi["height"])

            # Log.info(roi)
            frame = self.wd.capture_roi(roi)
            debug_frame = frame.copy()

            if self.findUIROI:
                fish_result = self.fd.detect(frame,self.wd.get_monitor())
                greenbar_result = self.gd.detect(frame)
                if greenbar_result.found and fish_result.found:

                    x, y, bw, bh = greenbar_result.greenbar_bbox
                    cx, cy = greenbar_result.greenbar_center

                    cv2.rectangle(debug_frame,(x, y),(x + bw, y + bh),(0, 255, 0),2,)
                    cv2.circle(debug_frame,(cx, cy),4,(0, 255, 0),-1,)

                    x, y, bw, bh = fish_result.fish_bbox
                    cx, cy = fish_result.fish_center
                    cv2.rectangle(debug_frame,(x, y),(x + bw, y + bh),(255, 0, 0),2,)
                    cv2.circle(debug_frame,(cx, cy),4,(255, 0, 0),-1,)                  
                    Log.debug(f"Fish Detector: {fish_result.fish_bbox} center: {fish_result.fish_center} name: {fish_result.template_name}" )

            # endTime = time.perf_counter()
            # # Log.info(endTime-startTime)
            cv2.imshow(preview_win, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
        cv2.destroyAllWindows()

    def update_monitor(self):
        m = self.wd.get_monitor()
        if not self.monitor == m:
            self.monitor = m
            self.is_update = True

    def update_ui(self):
        Log.info("detecting UI")
        self.is_update = False
        result = self.ud.detect(self.wd.capture(),self.wd.get_monitor())
        if result.found:
            Log.success(f"UI Detected with confidence: {result.confidence}")
            self.findUIROI = True
            self.roi = result.roi_monitor
        return result


    def _resize_preview_window(
        self,
        window_name: str,
        width: int,
        height: int,
    ) -> None:
        new_size = (
            int(width),
            int(height),
        )

        if self.last_preview_size == new_size:
            return

        cv2.resizeWindow(
            window_name,
            new_size[0],
            new_size[1],
        )

        self.last_preview_size = new_size



if __name__ == "__main__":
    App().run()




    
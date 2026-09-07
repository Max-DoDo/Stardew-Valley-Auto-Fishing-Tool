from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from src.detector.fish_detector import FishDetector
from src.detector.fish_y_detector import FishYDetector
from src.detector.process_bar_detector import ProcessBarDetector
from src.detector.greenbar_detector import GreenBarDetector
from src.model.q_model import QModel
from src.model.rule_model import RuleModel
from src.model.state import State
from src.tools.config_manager import ConfigManager
from src.tools.input_controller import InputController
from src.tools.log import Log
from src.detector.ui_detector import UIDetector
from src.detector.window_detector import WindowDetector
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

        self.last_fish_y = None
        self.last_greenbar_center_y = None

        self.wd = WindowDetector()
        self.ud = UIDetector()
        self.gd = GreenBarDetector()
        self.fd = FishDetector()
        self.pd = ProcessBarDetector()

        self.model = QModel()

        self.input = InputController()

        self.fyd = FishYDetector(
            hsv_low=(15, 50, 100),
            hsv_high=(45, 220, 255),
            fish_x1_ratio=0.25,
            fish_x2_ratio=0.75,
            min_row_score=3,
        )
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

            # Log.debug(roi)
            frame = self.wd.capture_roi(roi)
            debug_frame = frame.copy()

            if self.findUIROI:
                fish_result = self.fd.detect(frame)
                greenbar_result = self.gd.detect(frame)

                # is fishing
                if greenbar_result.found and fish_result.found:

                    x, y, bw, bh = greenbar_result.greenbar_bbox
                    cx, cy = greenbar_result.greenbar_center


                    cv2.rectangle(debug_frame,(x, y),(x + bw, y + bh),(0, 255, 0),2,)
                    cv2.circle(debug_frame,(cx, cy),4,(0, 255, 0),-1,)

                    cx, fishy = fish_result.fish_center

                    s = self.build_state(greenbar_result,fish_result)
                    self.model.learn(next_state=s, reward=self.compute_reward(s))
                    p = self.model.predict(s)
                    self.input.apply_action(p)
                    Log.info(s,p)

                    if y <= fish_result.fish_center[1] < y + bh:
                        cv2.circle(debug_frame,(cx, fishy),6,(255, 0, 0),-1,)
                    else:
                        cv2.circle(debug_frame,(cx, fishy),6,(0, 0, 255),-1,)
                else:
                    self.input.reset()
                    self.model.reset_episode()

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
        self.findUIROI = False
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

    def build_state(self,greenbar_result,fish_result ):
        if (
            not self.findUIROI
            or not fish_result.found
            or fish_result.fish_center is None
            or not greenbar_result.found
            or greenbar_result.greenbar_bbox is None
            or greenbar_result.greenbar_center is None
        ):
            self.last_fish_y = None
            self.last_greenbar_center_y = None

            return State(
                isFishing=False,
                fish_in_greenbar=False,
            )
        
        _, fish_y = fish_result.fish_center
        _, y, _, bh = greenbar_result.greenbar_bbox
        _, greenbar_center_y = greenbar_result.greenbar_center
        fish_in_greenbar = y <= fish_y < y + bh

        distance_y = fish_y - greenbar_center_y

        # build fish vel
        if self.last_fish_y is None:
            fish_velocity_y = 0
        else:
            fish_velocity_y = fish_y - self.last_fish_y
                
        # build greenbar vel
        if self.last_greenbar_center_y is None:
            greenbar_velocity_y = 0
        else:
            greenbar_velocity_y = greenbar_center_y - self.last_greenbar_center_y

        self.last_fish_y = fish_y
        self.last_greenbar_center_y = greenbar_center_y

        return State(
            isFishing=True,
            fish_in_greenbar=fish_in_greenbar,
            fish_y=fish_y,
            greenbar_center_y=greenbar_center_y,
            greenbar_height=bh,
            distance = distance_y,
            fish_velocity_y=fish_velocity_y,
            greenbar_velocity_y=greenbar_velocity_y,
        )

    def compute_reward(self, state: State) -> float:
        if not state.isFishing:
            return -2.0
        reward = 0.0

        if state.fish_in_greenbar:
            reward += 1.0
        else:
            reward -= 2.0

        return reward
if __name__ == "__main__":
    App().run()




    
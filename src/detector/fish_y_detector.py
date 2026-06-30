from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(slots=True)
class FishYResult:
    found: bool
    fish_y: Optional[int] = None
    fish_center: Optional[Point] = None
    confidence: float = 0.0
    max_row_score: int = 0


class FishYDetector:
    def __init__(
        self,
        hsv_low: tuple[int, int, int],
        hsv_high: tuple[int, int, int],
        fish_x1_ratio: float = 0.25,
        fish_x2_ratio: float = 0.75,
        min_row_score: int = 3,
        smooth_alpha: float = 0.35,
    ):
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)

        self.fish_x1_ratio = fish_x1_ratio
        self.fish_x2_ratio = fish_x2_ratio

        self.min_row_score = min_row_score
        self.smooth_alpha = smooth_alpha

        self.last_fish_y: Optional[int] = None

    def detect(
        self,
        frame: np.ndarray,
    ) -> FishYResult:
        frame_h, frame_w = frame.shape[:2]

        fish_x1 = int(frame_w * self.fish_x1_ratio)
        fish_x2 = int(frame_w * self.fish_x2_ratio)

        fish_lane = frame[:, fish_x1:fish_x2]

        hsv = cv2.cvtColor(
            fish_lane,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.inRange(
            hsv,
            self.hsv_low,
            self.hsv_high,
        )

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        row_scores = np.sum(
            mask > 0,
            axis=1,
        )

        max_row_score = int(np.max(row_scores))

        if max_row_score < self.min_row_score:
            return FishYResult(
                found=False,
                max_row_score=max_row_score,
            )

        strong_rows = np.where(
            row_scores >= max_row_score * 0.5,
        )[0]

        if len(strong_rows) == 0:
            return FishYResult(
                found=False,
                max_row_score=max_row_score,
            )

        raw_fish_y = int(np.mean(strong_rows))

        if self.last_fish_y is None:
            smooth_fish_y = raw_fish_y
        else:
            smooth_fish_y = int(
                self.last_fish_y * (1.0 - self.smooth_alpha)
                + raw_fish_y * self.smooth_alpha
            )

        self.last_fish_y = smooth_fish_y

        fish_x = int((fish_x1 + fish_x2) / 2)

        confidence = min(
            1.0,
            max_row_score / max(1, fish_x2 - fish_x1),
        )

        return FishYResult(
            found=True,
            fish_y=smooth_fish_y,
            fish_center=(fish_x, smooth_fish_y),
            confidence=confidence,
            max_row_score=max_row_score,
        )

    def draw_debug(
        self,
        frame: np.ndarray,
        result: FishYResult,
    ) -> np.ndarray:
        debug_frame = frame.copy()

        frame_h, frame_w = frame.shape[:2]

        fish_x1 = int(frame_w * self.fish_x1_ratio)
        fish_x2 = int(frame_w * self.fish_x2_ratio)

        cv2.rectangle(
            debug_frame,
            (fish_x1, 0),
            (fish_x2, frame_h),
            (255, 255, 0),
            1,
        )

        if result.found and result.fish_y is not None:
            cv2.line(
                debug_frame,
                (0, result.fish_y),
                (frame_w, result.fish_y),
                (255, 0, 0),
                2,
            )

            cv2.circle(
                debug_frame,
                result.fish_center,
                4,
                (255, 0, 0),
                -1,
            )

            cv2.putText(
                debug_frame,
                f"fish_y={result.fish_y} conf={result.confidence:.2f}",
                (10, max(20, result.fish_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return debug_frame
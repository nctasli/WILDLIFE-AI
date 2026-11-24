"""Automatic night-mode contrast enhancement using CLAHE."""

from __future__ import annotations

import logging
import cv2
import numpy as np

import config

logger = logging.getLogger("camera.enhancer")


class NightModeEnhancer:
    """Enhance low-light video frames using CLAHE on the luminance channel."""

    def __init__(
        self,
        *,
        auto: bool | None = None,
        enabled: bool | None = None,
        threshold: float | None = None,
        clip_limit: float | None = None,
        tile_grid_size: tuple[int, int] | None = None,
        draw_indicator: bool | None = None,
    ) -> None:
        self.auto = config.NIGHT_MODE_AUTO if auto is None else auto
        self.enabled = enabled  # None means evaluate dynamically via self.auto
        self.threshold = (
            config.NIGHT_MODE_THRESHOLD if threshold is None else threshold
        )
        self.clip_limit = (
            config.CLAHE_CLIP_LIMIT if clip_limit is None else clip_limit
        )
        self.tile_grid_size = (
            config.CLAHE_TILE_GRID_SIZE if tile_grid_size is None else tile_grid_size
        )
        self.draw_indicator = (
            config.NIGHT_MODE_DRAW_INDICATOR
            if draw_indicator is None
            else draw_indicator
        )
        self._clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=self.tile_grid_size,
        )

    def calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness (0-255) of the frame."""
        if frame.ndim == 2:
            return float(np.mean(frame))
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        return float(np.mean(frame))

    def should_enhance(self, brightness: float) -> bool:
        """Return True if CLAHE should be applied to the frame."""
        if self.enabled is not None:
            return self.enabled
        return self.auto and (brightness < self.threshold)

    def enhance(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float]:
        """Apply CLAHE enhancement if low-light condition is met.

        Returns (processed_frame, is_night_active, measured_brightness).
        """
        brightness = self.calculate_brightness(frame)
        if not self.should_enhance(brightness):
            return frame, False, brightness

        # Apply CLAHE to the Luminance (L) channel in LAB color space
        if frame.ndim == 3 and frame.shape[2] == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            enhanced_l = self._clahe.apply(l_channel)
            merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
            enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        elif frame.ndim == 2:
            enhanced_frame = self._clahe.apply(frame)
        else:
            enhanced_frame = frame

        if (
            self.draw_indicator
            and enhanced_frame.ndim >= 2
            and enhanced_frame.shape[1] >= 120
            and enhanced_frame.shape[0] >= 60
        ):
            # Draw subtle badge on top-left
            badge_text = f"NIGHT [CLAHE] B:{brightness:.0f}"
            cv2.putText(
                enhanced_frame,
                badge_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                enhanced_frame,
                badge_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return enhanced_frame, True, brightness


__all__ = ["NightModeEnhancer"]

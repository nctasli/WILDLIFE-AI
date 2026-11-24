from __future__ import annotations

import numpy as np
import pytest

from camera.enhancer import NightModeEnhancer


def test_enhancer_leaves_bright_frame_untouched() -> None:
    bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    enhancer = NightModeEnhancer(auto=True, threshold=65.0, draw_indicator=False)

    enhanced, is_night, brightness = enhancer.enhance(bright_frame)

    assert not is_night
    assert brightness == pytest.approx(180.0, abs=1.0)
    assert np.array_equal(enhanced, bright_frame)


def test_enhancer_boosts_contrast_on_dark_frame() -> None:
    # Frame with dark background (20) and faint object (35)
    dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    dark_frame[30:70, 30:70] = 35

    enhancer = NightModeEnhancer(auto=True, threshold=65.0, draw_indicator=False)
    enhanced, is_night, brightness = enhancer.enhance(dark_frame)

    assert is_night
    assert brightness < 65.0
    # Contrast difference (max - min) should increase with CLAHE
    orig_diff = int(np.max(dark_frame)) - int(np.min(dark_frame))
    enhanced_diff = int(np.max(enhanced)) - int(np.min(enhanced))
    assert enhanced_diff > orig_diff


def test_enhancer_manual_force_enabled() -> None:
    bright_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    enhancer = NightModeEnhancer(enabled=True, draw_indicator=False)

    enhanced, is_night, _ = enhancer.enhance(bright_frame)
    assert is_night


def test_enhancer_manual_force_disabled() -> None:
    dark_frame = np.full((100, 100, 3), 15, dtype=np.uint8)
    enhancer = NightModeEnhancer(enabled=False)

    enhanced, is_night, _ = enhancer.enhance(dark_frame)
    assert not is_night
    assert np.array_equal(enhanced, dark_frame)


def test_enhancer_handles_grayscale_frame() -> None:
    gray_dark = np.full((100, 100), 25, dtype=np.uint8)
    gray_dark[40:60, 40:60] = 40

    enhancer = NightModeEnhancer(auto=True, threshold=65.0, draw_indicator=False)
    enhanced, is_night, brightness = enhancer.enhance(gray_dark)

    assert is_night
    assert enhanced.shape == (100, 100)
    assert not np.array_equal(enhanced, gray_dark)


def test_enhancer_draws_badge_on_sufficiently_large_frame() -> None:
    dark_frame = np.full((120, 150, 3), 20, dtype=np.uint8)
    enhancer = NightModeEnhancer(auto=True, threshold=65.0, draw_indicator=True)

    enhanced, is_night, _ = enhancer.enhance(dark_frame)
    assert is_night
    # The badge draws yellow (0, 255, 255) pixels in BGR
    assert np.any(enhanced[:, :, 1] == 255)

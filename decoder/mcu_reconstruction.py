"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""

import numpy as np
from util import logger


def mcus_to_ycbcr_array(mcus_y: np.ndarray, mcus_cb: np.ndarray, mcus_cr: np.ndarray,
                        img_width: int, img_height: int) -> np.ndarray:
    """Convert MCU arrays back to YCbCr image array."""
    padded_height = ((img_height + 7) // 8) * 8
    padded_width = ((img_width + 7) // 8) * 8

    Y_channel = np.zeros((padded_height, padded_width))
    Cb_channel = np.zeros((padded_height, padded_width))
    Cr_channel = np.zeros((padded_height, padded_width))

    mcu_index = 0
    for row in range(0, padded_height, 8):
        for col in range(0, padded_width, 8):
            if mcu_index < len(mcus_y):
                Y_channel[row:row + 8, col:col + 8] = mcus_y[mcu_index]
                Cb_channel[row:row + 8, col:col + 8] = mcus_cb[mcu_index]
                Cr_channel[row:row + 8, col:col + 8] = mcus_cr[mcu_index]
                mcu_index += 1

    Y_channel = Y_channel[:img_height, :img_width]
    Cb_channel = Cb_channel[:img_height, :img_width]
    Cr_channel = Cr_channel[:img_height, :img_width]

    return np.stack([Y_channel, Cb_channel, Cr_channel], axis=-1)


def detect_errors(mcus_y: np.ndarray, mcus_cb: np.ndarray, mcus_cr: np.ndarray) -> None:
    """Detect and print suspicious values in MCUs."""
    for i, (mcu_y, mcu_cb, mcu_cr) in enumerate(zip(mcus_y, mcus_cb, mcus_cr)):
        for j, (component, name) in enumerate([(mcu_y, 'Y'), (mcu_cb, 'Cb'), (mcu_cr, 'Cr')]):
            if np.any(component > 255) or np.any(component < 0):
                logger.debug(f"Warning: MCU {i}, component {j} ({name}) has values outside [0, 255]")

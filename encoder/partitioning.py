"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


def partition(channel: np.ndarray) -> np.ndarray:
    """Partition channel into 8x8 MCUs with edge padding."""
    h, w = channel.shape
    padded_h = ((h + 7) // 8) * 8
    padded_w = ((w + 7) // 8) * 8
    padded = np.pad(channel, ((0, padded_h - h), (0, padded_w - w)), mode="edge")
    blocks = (
        padded.reshape(padded_h // 8, 8, padded_w // 8, 8)
        .swapaxes(1, 2)
        .reshape(-1, 8, 8)
        .astype(np.int16)
    )
    return blocks

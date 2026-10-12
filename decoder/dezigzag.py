"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


def dezigzag(array: np.ndarray) -> np.ndarray:
    """Reverse zigzag ordering to reconstruct 8x8 blocks."""
    order = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ], dtype=np.int8)
    num_blocks = array.shape[0]
    result = np.empty((num_blocks, 8, 8), dtype=array.dtype)
    result_flat = result.reshape(num_blocks, 64)
    result_flat[:, :] = array[:, order.ravel()]
    return result

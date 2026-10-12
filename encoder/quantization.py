"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


def quantize(mcu_array: np.ndarray, quantization_table: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients using quantization table."""
    x = mcu_array / quantization_table
    q = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return q.astype(np.int16)

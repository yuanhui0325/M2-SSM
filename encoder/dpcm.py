"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np
from typing import List


def dpcm_encode(dc_coefficients: np.ndarray) -> List[int]:
    """Apply DPCM encoding to DC coefficients."""
    dc = np.asarray(dc_coefficients, dtype=np.int32)
    result = np.zeros(len(dc), dtype=np.int32)
    result[0] = dc[0]
    result[1:] = dc[1:] - dc[:-1]
    return result.tolist()

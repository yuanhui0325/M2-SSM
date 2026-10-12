"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import List
import numpy as np


def dpcm_decode(dpcm_coefficients: List[int]) -> np.ndarray:
    """Decode DPCM-encoded DC coefficients."""
    return np.cumsum(dpcm_coefficients)

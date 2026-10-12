"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""

import numpy as np
from scipy.fft import idct


def IDCT(dct_MCUs: np.ndarray) -> np.ndarray:
    """Apply Inverse Discrete Cosine Transform to MCUs."""
    idct_MCUs = np.zeros_like(dct_MCUs)
    for i, MCU in enumerate(dct_MCUs):
        for j, component in enumerate(MCU):
            idct_MCUs[i, j] = idct(idct(component, axis=0, norm='ortho'),
                                   axis=1, norm='ortho')
    return idct_MCUs

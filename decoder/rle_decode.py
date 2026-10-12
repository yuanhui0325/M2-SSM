"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import List, Tuple
import numpy as np


def rle_decode(ac_tuples: List[Tuple[int, int]]) -> np.ndarray:
    """Decode RLE-encoded AC coefficients for a single MCU."""
    ac = []
    for zeros, value in ac_tuples:
        if (zeros, value) == (0, 0):
            break
        ac.extend([0] * zeros)
        ac.append(value)
    while len(ac) < 63:
        ac.append(0)
    return np.array(ac[:63])


def rle_decode_mcus(rle_data: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Decode RLE-encoded AC coefficients for multiple MCUs."""
    decoded_acs = []
    for ac_tuples in rle_data:
        decoded_ac = rle_decode(ac_tuples)
        decoded_acs.append(decoded_ac)
    return np.array(decoded_acs)

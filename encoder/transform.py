"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np
from . import transform_cy


def transform(MCU_list: np.ndarray) -> np.ndarray:
    """Apply DCT transform to MCU blocks using Cython.

    Converts input blocks from spatial domain to DCT domain.
    """
    return transform_cy.transform(MCU_list)

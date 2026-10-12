"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""

import numpy as np


def dequantize(quant_y: np.ndarray,
               quant_cb: np.ndarray,
               quant_cr: np.ndarray,
               quantization_table_lum: np.ndarray,
               quantization_table_chrom: np.ndarray) -> tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
    """Reverse the quantization process."""
    dequant_y = quant_y * quantization_table_lum
    dequant_cb = quant_cb * quantization_table_chrom
    dequant_cr = quant_cr * quantization_table_chrom
    return dequant_y, dequant_cb, dequant_cr

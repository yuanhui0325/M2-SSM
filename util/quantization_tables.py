"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


# Quantization tables defined by the JPEG standard (Annex K.1, p. 143) https://www.w3.org/Graphics/JPEG/itu-t81.pdf
quantization_table_lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]])

quantization_table_chrom = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                     [18, 21, 26, 66, 99, 99, 99, 99],
                                     [24, 26, 56, 99, 99, 99, 99, 99],
                                     [47, 66, 99, 99, 99, 99, 99, 99],
                                     [99, 99, 99, 99, 99, 99, 99, 99],
                                     [99, 99, 99, 99, 99, 99, 99, 99],
                                     [99, 99, 99, 99, 99, 99, 99, 99],
                                     [99, 99, 99, 99, 99, 99, 99, 99]])


# --- Quantization table for experiments ---

# val = 200

# quantization_table_lum = np.array([[val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val],
#                                    [val, val, val, val, val, val, val, val]])

# quantization_table_chrom = quantization_table_lum

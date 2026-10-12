"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


def rgb_to_ycbcr(pixel_array: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCbCr color space (JPEG standard)."""
    conv_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    rgb_float = pixel_array.astype(np.float64)
    h, w, c = rgb_float.shape
    rgb_flat = rgb_float.reshape(-1, 3)

    # Vectorized conversion: Y/Cb/Cr = rgb @ matrix^T + offset
    ycbcr_flat = rgb_flat @ conv_matrix.T + np.array([0, 128, 128])

    ycbcr = ycbcr_flat.reshape(h, w, 3).astype(np.uint8)
    return ycbcr


def ycbcr_to_rgb(ycbcr_array: np.ndarray) -> np.ndarray:
    """Convert YCbCr image back to RGB color space."""
    ycbcr_centered = ycbcr_array.copy().astype(np.float64)
    ycbcr_centered[:, :, 1] -= 128  # Center Cb around 0
    ycbcr_centered[:, :, 2] -= 128  # Center Cr around 0

    # Inverse conversion matrix
    conversion_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.34414, -0.71414],
        [1.0, 1.772, 0.0]
    ])

    rgb_image = np.dot(ycbcr_centered, conversion_matrix.T)
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image

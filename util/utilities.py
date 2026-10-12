"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np


def print_9_mcus(mcu_list: np.ndarray, description: str, flattened: bool = False) -> None:
    """
    Print the first 9 MCUs for debugging.

    Args:
        mcu_list: Array of MCUs
        description: Description of the MCU processing step
        flattened: Whether the MCUs are flattened (1D) or 2D
    """
    print("\n\n")
    print(description)
    if not flattened:
        print(type(mcu_list), mcu_list.shape, type(mcu_list[0][0][0]))
    else:
        print(type(mcu_list), mcu_list.shape, type(mcu_list[0][0]))
    for i in range(min(9, len(mcu_list))):
        print(mcu_list[i])
        print()


def print_3_mcus(mcu_list: np.ndarray, description: str, flattened: bool = False) -> None:
    """
    Print the first 3 MCUs for debugging.

    Args:
        mcu_list: Array of MCUs
        description: Description of the MCU processing step
        flattened: Whether the MCUs are flattened (1D) or 2D
    """
    print("\n")
    print(description)
    if not flattened:
        print(type(mcu_list), mcu_list.shape, type(mcu_list[0][0][0]))
    else:
        print(type(mcu_list), mcu_list.shape, type(mcu_list[0][0]))
    for i in range(min(3, len(mcu_list))):
        print(mcu_list[i])
        print()


def print_3x3_mcus(
    y_mcus: np.ndarray,
    cb_mcus: np.ndarray,
    cr_mcus: np.ndarray,
    description: str = "",
    flattened: bool = False
) -> None:
    """
    Print the first 3 MCUs for each color component (Y, Cb, Cr).

    Args:
        y_mcus: Array of Y (luminance) MCUs
        cb_mcus: Array of Cb (chrominance blue) MCUs
        cr_mcus: Array of Cr (chrominance red) MCUs
        description: Description of the MCU processing step
        flattened: Whether the MCUs are flattened (1D) or 2D
    """
    print()
    mcus = [y_mcus, cb_mcus, cr_mcus]
    mcu_names = ["Y_MCUs", "Cb_MCUs", "Cr_MCUs"]

    for i in range(3):
        print("\n")
        print(f"First 3 {mcu_names[i]} after {description}")
        if not flattened:
            print(type(mcus[i]), mcus[i].shape, type(mcus[i][0][0][0]))
        else:
            print(type(mcus[i]), mcus[i].shape, type(mcus[i][0][0]))
        for j in range(min(3, len(mcus[i]))):
            print(mcus[i][j])
            print()

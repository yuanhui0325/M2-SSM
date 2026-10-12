"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import List
from bitstring import BitArray


def interleave_mcus(
    encoded_dc_y: List[BitArray], encoded_ac_y: List[List[BitArray]],
    encoded_dc_cb: List[BitArray], encoded_ac_cb: List[List[BitArray]],
    encoded_dc_cr: List[BitArray], encoded_ac_cr: List[List[BitArray]]
) -> BitArray:
    """Interleave Y, Cb, Cr MCUs in 4:4:4 format."""
    result = BitArray()
    num_mcus = len(encoded_dc_y)

    for i in range(num_mcus):
        result += encoded_dc_y[i]
        for ac_bitarray in encoded_ac_y[i]:
            result += ac_bitarray

        result += encoded_dc_cb[i]
        for ac_bitarray in encoded_ac_cb[i]:
            result += ac_bitarray

        result += encoded_dc_cr[i]
        for ac_bitarray in encoded_ac_cr[i]:
            result += ac_bitarray

    return result

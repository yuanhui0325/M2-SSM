"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import List
from bitstring import BitArray

from util.bit_utils import bits_required, value_to_bits
from .huffman import huffman_encode


def encode_dc_coefficients(dpcm: List[int], huffman_table) -> list[BitArray]:
    """Huffman encode DC coefficients."""
    encoded_dc = []
    for dc_value in dpcm:
        category = bits_required(dc_value)
        huffman_code = huffman_encode(category, huffman_table)
        dc_bits = value_to_bits(dc_value)
        encoded_dc.append(huffman_code + dc_bits)
    return encoded_dc

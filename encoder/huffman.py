"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from bitstring import BitArray
from typing import Dict


def huffman_encode(symbol: int, huffman_table: Dict[int, str]) -> BitArray:
    """Look up Huffman code for a symbol."""
    return BitArray("0b" + huffman_table[symbol])

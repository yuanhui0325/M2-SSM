# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython-optimized AC coefficient encoding
Matches encode_ac.py logic but with Cython speed optimizations
"""

from typing import List, Tuple
from bitstring import BitArray
from .huffman import huffman_encode


cdef inline int _bits_required_fast(int v):
    """JPEG 'size' = Anzahl Bits der Betragdarstellung. 0 -> 0."""
    if v == 0:
        return 0
    cdef int abs_v = abs(v)
    return abs_v.bit_length()


def _build_huff_cache(huffman_table):
    """
    Cache: symbol -> BitArray
    Nur für Symbole, die in der Tabelle existieren.
    """
    return {sym: huffman_encode(sym, huffman_table) for sym in huffman_table.keys()}


def encode_ac_coefficients(
    rle: List[List[Tuple[int, int]]],
    huffman_table
) -> List[List[BitArray]]:
    """
    Cython-optimized version of encode_ac_coefficients.
    Takes RLE data (list of MCUs with (zero_run, ac_value) tuples)
    Returns List[List[BitArray]] - encoded AC coefficients
    """
    # Huffman-Codes einmal vorberechnen
    huff_code = _build_huff_cache(huffman_table)

    encoded_ac: List[List[BitArray]] = []

    for mcu_rle in rle:
        encoded_mcu: List[BitArray] = []

        for zero_run, ac_value in mcu_rle:
            size = _bits_required_fast(ac_value)
            symbol = (zero_run << 4) | size  # RRRRSSSS

            # BitArray erstellen und append (schneller als zu concat)
            out = BitArray()
            out.append(huff_code[symbol])

            if size:
                amp = (ac_value if ac_value > 0 else (1 << size) - 1 + ac_value)
                out.append(f"uint:{size}={amp}")

            encoded_mcu.append(out)

        encoded_ac.append(encoded_mcu)

    return encoded_ac

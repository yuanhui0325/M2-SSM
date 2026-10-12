"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import List, Tuple, Dict
from bitstring import BitArray

from util import huffman_tables

# Cache reverse tables for performance
_DC_Y_REV = {v: k for k, v in huffman_tables.DC_Y.items()}
_DC_CBCR_REV = {v: k for k, v in huffman_tables.DC_CbCr.items()}
_AC_Y_REV = {v: k for k, v in huffman_tables.AC_Y.items()}
_AC_CBCR_REV = {v: k for k, v in huffman_tables.AC_CbCr.items()}


def _decode_dc_value(bitstream: BitArray, reverse_table: Dict, pos: int) -> Tuple[int, int]:
    """Internal DC value decoder with cached reverse table."""
    cdef int size = 0
    cdef int value = 0
    cdef int start_pos = pos
    cdef str code_str = ""

    while pos < len(bitstream):
        code_str += "1" if bitstream[pos] else "0"
        if code_str in reverse_table:
            size = reverse_table[code_str]

            if size == 0:
                return 0, pos + 1

            value_bits = bitstream[pos + 1:pos + 1 + size]
            pos += 1 + size

            value = int(value_bits.bin, 2)

            if not value_bits[0]:
                value = value - (1 << size) + 1

            return value, pos

        pos += 1

    raise ValueError(f"Could not decode DC value starting at position {start_pos}")


def _decode_ac_tuple(bitstream: BitArray, reverse_table: Dict, pos: int) -> Tuple[Tuple[int, int], int]:
    """Internal AC tuple decoder with cached reverse table."""
    cdef int run_size = 0
    cdef int run = 0
    cdef int size = 0
    cdef int value = 0
    cdef int start_pos = pos
    cdef str code_str = ""

    while pos < len(bitstream):
        code_str += "1" if bitstream[pos] else "0"
        if code_str in reverse_table:
            run_size = reverse_table[code_str]

            if run_size == 0:
                return (0, 0), pos + 1

            run = run_size >> 4
            size = run_size & 0x0F

            if size == 0:
                if run == 15:
                    return (15, 0), pos + 1
                else:
                    raise ValueError(f"Invalid AC coefficient: run={run}, size=0")

            value_bits = bitstream[pos + 1:pos + 1 + size]
            pos += 1 + size

            value = int(value_bits.bin, 2)

            if not value_bits[0]:
                value = value - (1 << size) + 1

            return (run, value), pos

        pos += 1

    raise ValueError(f"Could not decode AC tuple starting at position {start_pos}")


def decode_dc_value(bitstream: BitArray, huffman_table: Dict[int, str], pos: int) -> Tuple[int, int]:
    """Decode a single DC coefficient from the bitstream."""
    cdef int size = 0
    cdef int value = 0
    cdef int start_pos = pos
    cdef str code_str = ""

    reverse_table = {v: k for k, v in huffman_table.items()}

    while pos < len(bitstream):
        code_str += "1" if bitstream[pos] else "0"
        if code_str in reverse_table:
            size = reverse_table[code_str]

            if size == 0:
                return 0, pos + 1

            value_bits = bitstream[pos + 1:pos + 1 + size]
            pos += 1 + size

            value = int(value_bits.bin, 2)

            if not value_bits[0]:
                value = value - (1 << size) + 1

            return value, pos

        pos += 1

    raise ValueError(f"Could not decode DC value starting at position {start_pos}")


def decode_ac_tuple(bitstream: BitArray, huffman_table: Dict[int, str], pos: int) -> Tuple[Tuple[int, int], int]:
    """Decode a single AC coefficient tuple from the bitstream."""
    cdef int run_size = 0
    cdef int run = 0
    cdef int size = 0
    cdef int value = 0
    cdef int start_pos = pos
    cdef str code_str = ""

    reverse_table = {v: k for k, v in huffman_table.items()}

    while pos < len(bitstream):
        code_str += "1" if bitstream[pos] else "0"
        if code_str in reverse_table:
            run_size = reverse_table[code_str]

            if run_size == 0:
                return (0, 0), pos + 1

            run = run_size >> 4
            size = run_size & 0x0F

            if size == 0:
                if run == 15:
                    return (15, 0), pos + 1
                else:
                    raise ValueError(f"Invalid AC coefficient: run={run}, size=0")

            value_bits = bitstream[pos + 1:pos + 1 + size]
            pos += 1 + size

            value = int(value_bits.bin, 2)

            if not value_bits[0]:
                value = value - (1 << size) + 1

            return (run, value), pos

        pos += 1

    raise ValueError(f"Could not decode AC tuple starting at position {start_pos}")


def deinterleave(
    huffman_bitstream: BitArray,
    num_mcus: int
) -> Tuple[List[BitArray], List[BitArray], List[BitArray], List[BitArray], List[BitArray], List[BitArray]]:
    """Deinterleave bitstream into separate DC and AC bitstreams for each MCU."""
    encoded_dc_y = []
    encoded_dc_cb = []
    encoded_dc_cr = []
    encoded_ac_y = []
    encoded_ac_cb = []
    encoded_ac_cr = []

    cdef int pos = 0
    cdef int mcu_idx = 0
    cdef int start_pos = 0
    cdef tuple ac_tuple

    for mcu_idx in range(num_mcus):
        start_pos = pos
        dc_value, pos = _decode_dc_value(huffman_bitstream, _DC_Y_REV, pos)
        encoded_dc_y.append(huffman_bitstream[start_pos:pos])

        start_pos = pos
        while True:
            ac_tuple, pos = _decode_ac_tuple(huffman_bitstream, _AC_Y_REV, pos)
            if ac_tuple == (0, 0):
                break
        encoded_ac_y.append(huffman_bitstream[start_pos:pos])

        start_pos = pos
        dc_value, pos = _decode_dc_value(huffman_bitstream, _DC_CBCR_REV, pos)
        encoded_dc_cb.append(huffman_bitstream[start_pos:pos])

        start_pos = pos
        while True:
            ac_tuple, pos = _decode_ac_tuple(huffman_bitstream, _AC_CBCR_REV, pos)
            if ac_tuple == (0, 0):
                break
        encoded_ac_cb.append(huffman_bitstream[start_pos:pos])

        start_pos = pos
        dc_value, pos = _decode_dc_value(huffman_bitstream, _DC_CBCR_REV, pos)
        encoded_dc_cr.append(huffman_bitstream[start_pos:pos])

        start_pos = pos
        while True:
            ac_tuple, pos = _decode_ac_tuple(huffman_bitstream, _AC_CBCR_REV, pos)
            if ac_tuple == (0, 0):
                break
        encoded_ac_cr.append(huffman_bitstream[start_pos:pos])

    return encoded_dc_y, encoded_dc_cb, encoded_dc_cr, encoded_ac_y, encoded_ac_cb, encoded_ac_cr


def huffman_decode_dc(
    encoded_dc_list: List[BitArray],
    huffman_table: Dict[int, str]
) -> List[int]:
    """Huffman decode a list of DC coefficients."""
    dpcm_values = []
    for encoded_dc in encoded_dc_list:
        dc_value, _ = decode_dc_value(encoded_dc, huffman_table, 0)
        dpcm_values.append(dc_value)
    return dpcm_values


def huffman_decode_ac(
    encoded_ac_list: List,
    huffman_table: Dict[int, str]
) -> List[List[Tuple[int, int]]]:
    """Huffman decode a list of AC coefficients."""
    rle_list = []
    for encoded_ac in encoded_ac_list:
        if isinstance(encoded_ac, list):
            concatenated = BitArray()
            for bit_array in encoded_ac:
                concatenated += bit_array
            encoded_ac = concatenated

        ac_tuples = []
        pos = 0
        while pos < len(encoded_ac):
            ac_tuple, pos = decode_ac_tuple(encoded_ac, huffman_table, pos)
            ac_tuples.append(ac_tuple)
            if ac_tuple == (0, 0):
                break
        rle_list.append(ac_tuples)
    return rle_list

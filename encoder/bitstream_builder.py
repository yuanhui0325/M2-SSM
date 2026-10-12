"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen

JPEG bitstream builder module.

References:
- https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format
- https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/
"""
import sys
from typing import Dict
import numpy as np
from bitstring import BitArray

from util import logger
from util.bit_utils import fill_up_last_byte, add_FF00
from .zigzag import zigzag


def build_header() -> bytes:
    result = bytes()
    result += bytes.fromhex("FF D8")
    result += bytes.fromhex("FF E0 00 10 4A 46 49 46 00")
    result += bytes.fromhex("01 01 01 00 48 00 48 00 00")
    return result


def build_quantization_table(table: np.ndarray, table_type: str) -> bytes:
    result = bytes()
    result += bytes.fromhex("FF DB 00 43")

    if table_type == "lum":
        result += bytes.fromhex("00")
    elif table_type == "chrom":
        result += bytes.fromhex("01")
    else:
        sys.exit(f"Error: quantization_table type {table_type} does not exist!")

    zz = zigzag(table.reshape(1, 8, 8))[0]
    for val in zz:
        result += int(val).to_bytes(1, "big")

    return result


def build_start_of_frame(
    color_depth: int,
    image_height: int,
    image_width: int,
    color_components: int,
    sub_sample_mode: str
) -> bytes:
    result = bytes()
    result += bytes.fromhex("FF C0 00 11")
    result += color_depth.to_bytes(1, "big")
    result += image_height.to_bytes(2, "big")
    result += image_width.to_bytes(2, "big")
    result += color_components.to_bytes(1, "big")

    if sub_sample_mode == "4:4:4":
        result += bytes.fromhex("01 11 00")
        result += bytes.fromhex("02 11 01")
        result += bytes.fromhex("03 11 01")
    elif sub_sample_mode == "4:2:0":
        sys.exit("SOF for 4:2:0 is not yet implemented!")
    else:
        sys.exit(f"Error: sub_sampling_mode {sub_sample_mode} is not supported!")

    return result


def build_huffman_tables(huff_tables: Dict) -> bytes:
    def segment_length(table):
        return 2 + 1 + 16 + len(table)

    ht_info_LUT = {
        "DC_Y": ("0", "0"),
        "AC_Y": ("1", "0"),
        "DC_CbCr": ("0", "1"),
        "AC_CbCr": ("1", "1")
    }

    def ht_info(tbl_ctr, table):
        result = BitArray()
        result.append("uint:4=" + ht_info_LUT[table][0])
        result.append("uint:4=" + ht_info_LUT[table][1])
        return result.bytes

    def number_of_huff_codes(tbl):
        result = bytes()
        for length in range(1, 17):
            total_nr = 0
            for key, value in tbl.items():
                if length == len(value):
                    total_nr += 1
            result += total_nr.to_bytes(1, "big")
        return result

    def symbols(tbl):
        result = bytes()
        for length in range(1, 17):
            for key, value in tbl.items():
                if len(value) == length:
                    result += key.to_bytes(1, "big")
        return result

    result = bytes()
    for tbl_ctr, table in enumerate(huff_tables):
        result += bytes.fromhex("FF C4")
        result += segment_length(huff_tables[table]).to_bytes(2, "big")
        result += ht_info(tbl_ctr, table)
        result += number_of_huff_codes(huff_tables[table])
        result += symbols(huff_tables[table])

    return result


def build_start_of_scan(color_components: int) -> bytes:
    result = bytes()
    result += bytes.fromhex("FF DA 00 0C")
    result += color_components.to_bytes(1, "big")
    result += bytes.fromhex("01 00")
    result += bytes.fromhex("02 11")
    result += bytes.fromhex("03 11")
    result += bytes.fromhex("00 3F 00")
    return result


def build_image_data(huffman_scan_bytes: bytes) -> bytes:
    return huffman_scan_bytes


def build_end_of_image() -> bytes:
    return bytes.fromhex("FF D9")


def build_bitstream(
    quantization_table_lum: np.ndarray,
    quantization_table_chrom: np.ndarray,
    image_height: int,
    image_width: int,
    huff_tables: Dict,
    huffman_scan_bytes: bytes
) -> bytes:
    color_depth = 8
    num_color_components = 3
    sub_sample_mode = "4:4:4"

    bytestream = bytes()
    bytestream += build_header()
    bytestream += build_quantization_table(quantization_table_lum, "lum")
    bytestream += build_quantization_table(quantization_table_chrom, "chrom")
    bytestream += build_start_of_frame(
        color_depth, image_height, image_width, num_color_components, sub_sample_mode
    )
    bytestream += build_huffman_tables(huff_tables)
    bytestream += build_start_of_scan(num_color_components)
    bytestream += build_image_data(huffman_scan_bytes)
    bytestream += build_end_of_image()

    return bytestream

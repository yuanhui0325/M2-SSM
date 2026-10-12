"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from typing import Tuple
from bitstring import BitArray
import numpy as np

from util import logger
from decoder.dezigzag import dezigzag


def remove_FF00_stuffing(image_bytes: bytes) -> bytearray:
    """Remove byte stuffing (0x00 after 0xFF) from image data."""
    result = bytearray()
    i = 0
    while i < len(image_bytes):
        result.append(image_bytes[i])
        if image_bytes[i] == 0xFF and i + 1 < len(image_bytes) and image_bytes[i + 1] == 0x00:
            i += 2
        else:
            i += 1
    return result


def parse_jpeg_bitstream(
    jpeg_bitstream: bytes
) -> Tuple[BitArray, np.ndarray, np.ndarray]:
    """Parse JPEG bitstream to extract Huffman data and quantization tables."""
    pos = 0
    quant_lum = None
    quant_chrom = None
    image_data = None

    if jpeg_bitstream[pos:pos + 2] != bytes.fromhex("FF D8"):
        raise ValueError("Invalid JPEG: Missing SOI marker")
    pos += 2

    while pos < len(jpeg_bitstream) - 2:
        if jpeg_bitstream[pos] != 0xFF:
            raise ValueError(f"Expected marker at position {pos}")

        marker = jpeg_bitstream[pos + 1]
        pos += 2

        if marker == 0xD9:
            break

        if marker == 0xDA:
            length = int.from_bytes(jpeg_bitstream[pos:pos + 2], 'big')
            pos += length
            eoi_pos = jpeg_bitstream.find(bytes.fromhex("FF D9"), pos)
            if eoi_pos == -1:
                raise ValueError("No EOI marker found")
            image_data = jpeg_bitstream[pos:eoi_pos]
            break

        if marker == 0xDB:
            length = int.from_bytes(jpeg_bitstream[pos:pos + 2], 'big')
            pos += 2
            table_info = jpeg_bitstream[pos]
            table_id = table_info & 0x0F
            pos += 1
            table_data = jpeg_bitstream[pos:pos + 64]
            pos += 64
            table_array = np.frombuffer(table_data, dtype=np.uint8).astype(np.int32)
            table_8x8 = dezigzag(np.expand_dims(table_array, axis=0))
            if table_id == 0:
                quant_lum = table_8x8
            elif table_id == 1:
                quant_chrom = table_8x8
            continue

        if marker not in [0xD8, 0xD9, 0x01]:
            length = int.from_bytes(jpeg_bitstream[pos:pos + 2], 'big')
            pos += length

    if image_data is None:
        raise ValueError("No image data found in JPEG")
    if quant_lum is None or quant_chrom is None:
        raise ValueError("Missing quantization tables")

    image_data_unstuffed = remove_FF00_stuffing(image_data)
    huffman_bitstream = BitArray(bytes=bytes(image_data_unstuffed))

    return huffman_bitstream, quant_lum, quant_chrom

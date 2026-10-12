"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class EncodingResult:
    """
    Container for JPEG encoding results at different stages.

    Depending on the encoding stage, different attributes will be populated.
    All stages return this object, but only the relevant data for that stage
    and all previous stages will be set.

    Encoding Pipeline Stages:
        1. Partition into MCUs (8x8 blocks)
        2. DCT transform (frequency domain)
        3. Quantization
        4. Extract DC coefficients
        5. Zigzag AC coefficients only
        6. DPCM encoding for DC
        7. RLE encoding for AC
        8. Huffman encoding
        9. Write JPEG file

    Attributes:
        img_width: Original image width in pixels
        img_height: Original image height in pixels

        # Stage 1: MCUs (8x8 blocks in YCbCr)
        mcus_y: Y component MCUs, shape (num_mcus, 8, 8)
        mcus_cb: Cb component MCUs, shape (num_mcus, 8, 8)
        mcus_cr: Cr component MCUs, shape (num_mcus, 8, 8)

        # Stage 2: DCT (frequency domain)
        dct_y: DCT-transformed Y MCUs, shape (num_mcus, 8, 8)
        dct_cb: DCT-transformed Cb MCUs, shape (num_mcus, 8, 8)
        dct_cr: DCT-transformed Cr MCUs, shape (num_mcus, 8, 8)

        # Stage 3: Quantization
        quant_y: Quantized Y MCUs, shape (num_mcus, 8, 8)
        quant_cb: Quantized Cb MCUs, shape (num_mcus, 8, 8)
        quant_cr: Quantized Cr MCUs, shape (num_mcus, 8, 8)
        quantization_table_lum: Quantization table for Y (luminance)
        quantization_table_chrom: Quantization table for Cb/Cr (chrominance)

        # Stage 4: DC Extraction
        dc_y: Extracted DC coefficients for Y, shape (num_mcus,)
        dc_cb: Extracted DC coefficients for Cb, shape (num_mcus,)
        dc_cr: Extracted DC coefficients for Cr, shape (num_mcus,)

        # Stage 5: AC Zigzag (only AC coefficients, 63 per MCU)
        ac_y: Zigzagged AC coefficients for Y, shape (num_mcus, 63)
        ac_cb: Zigzagged AC coefficients for Cb, shape (num_mcus, 63)
        ac_cr: Zigzagged AC coefficients for Cr, shape (num_mcus, 63)

        # Stage 6: DPCM (DC differential encoding)
        dpcm_y: DPCM-encoded DC differences for Y
        dpcm_cb: DPCM-encoded DC differences for Cb
        dpcm_cr: DPCM-encoded DC differences for Cr

        # Stage 7: RLE (AC run-length encoding)
        rle_y: RLE-encoded AC tuples for Y
        rle_cb: RLE-encoded AC tuples for Cb
        rle_cr: RLE-encoded AC tuples for Cr

        # Stage 8: Huffman encoding (DC coefficients)
        encoded_dc_y: Huffman-encoded DC coefficients for Y
        encoded_dc_cb: Huffman-encoded DC coefficients for Cb
        encoded_dc_cr: Huffman-encoded DC coefficients for Cr

        # Stage 8: Huffman encoding (AC coefficients)
        encoded_ac_y: Huffman-encoded AC coefficients for Y
        encoded_ac_cb: Huffman-encoded AC coefficients for Cb
        encoded_ac_cr: Huffman-encoded AC coefficients for Cr

        # Stage 9: Huffman encoding (interleaved bitstream)
        huffman_bitstream: Huffman-encoded bitstream (BitArray)

        # Stage 10: JPEG file
        jpeg_bitstream: Final JPEG file bytes
        output_file: Path where JPEG file was written
    """
    img_width: int
    img_height: int

    # MCUs stage
    mcus_y: Optional[np.ndarray] = None
    mcus_cb: Optional[np.ndarray] = None
    mcus_cr: Optional[np.ndarray] = None

    # DCT stage
    dct_y: Optional[np.ndarray] = None
    dct_cb: Optional[np.ndarray] = None
    dct_cr: Optional[np.ndarray] = None

    # Quantization stage
    quant_y: Optional[np.ndarray] = None
    quant_cb: Optional[np.ndarray] = None
    quant_cr: Optional[np.ndarray] = None
    quantization_table_lum: Optional[np.ndarray] = None  # Quantization table for Y (luminance)
    quantization_table_chrom: Optional[np.ndarray] = None  # Quantization table for Cb/Cr (chrominance)

    # DC extraction stage
    dc_y: Optional[np.ndarray] = None
    dc_cb: Optional[np.ndarray] = None
    dc_cr: Optional[np.ndarray] = None

    # AC zigzag stage
    ac_y: Optional[np.ndarray] = None
    ac_cb: Optional[np.ndarray] = None
    ac_cr: Optional[np.ndarray] = None

    # DPCM stage (DC coefficients)
    dpcm_y: Optional[List] = None
    dpcm_cb: Optional[List] = None
    dpcm_cr: Optional[List] = None

    # RLE stage (AC coefficients)
    rle_y: Optional[List] = None
    rle_cb: Optional[List] = None
    rle_cr: Optional[List] = None

    # Huffman encoding stage (DC coefficients)
    encoded_dc_y: Optional[List] = None
    encoded_dc_cb: Optional[List] = None
    encoded_dc_cr: Optional[List] = None

    # Huffman encoding stage (AC coefficients)
    encoded_ac_y: Optional[List] = None
    encoded_ac_cb: Optional[List] = None
    encoded_ac_cr: Optional[List] = None

    # Huffman stage
    huffman_bitstream: Optional[object] = None  # BitArray
    jpeg_bitstream: Optional[bytes] = None
    output_file: Optional[str] = None

"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""

import numpy as np

from util import EncodingResult, logger, huffman_tables
from util.encoding_stages import (
    STAGE_JPEG, STAGE_INTERLEAVER, STAGE_AC, STAGE_DC, STAGE_RLE, STAGE_DPCM,
    STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS
)
from .dpcm_decode import dpcm_decode
from .rle_decode import rle_decode_mcus
from .dezigzag import dezigzag
from .quantization_decode import dequantize
from .idct import IDCT
from .mcu_reconstruction import mcus_to_ycbcr_array, detect_errors
from .huffman_decode import deinterleave, huffman_decode_dc, huffman_decode_ac
from .jpeg_parser import parse_jpeg_bitstream


def decode(encoding_result: EncodingResult, last_encoding_stage: str) -> np.ndarray:
    """Decode JPEG-encoded data back to YCbCr image array."""
    img_width = encoding_result.img_width
    img_height = encoding_result.img_height

    logger.info(f"Decoder is running in '{last_encoding_stage}' mode")

    # Initialize all variables
    huffman_bitstream = None
    dpcm_y = dpcm_cb = dpcm_cr = None
    rle_y = rle_cb = rle_cr = None
    dc_y = dc_cb = dc_cr = None
    ac_y = ac_cb = ac_cr = None
    quant_y = quant_cb = quant_cr = None
    dct_y = dct_cb = dct_cr = None
    mcus_y = mcus_cb = mcus_cr = None

    mcu_cols = (img_width + 7) // 8
    mcu_rows = (img_height + 7) // 8
    num_mcus = mcu_cols * mcu_rows

    # Define which modes skip which steps (steps that weren't performed by encoder)
    # If the mode is in the list, the step was NOT performed by the encoder, so skip it
    SKIP_JPEG_PARSING = [STAGE_INTERLEAVER, STAGE_AC, STAGE_DC, STAGE_RLE, STAGE_DPCM,
                         STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_DEINTERLEAVE = [STAGE_AC, STAGE_DC, STAGE_RLE, STAGE_DPCM,
                         STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_HUFFMAN_AC_DECODE = [STAGE_DC, STAGE_RLE, STAGE_DPCM, STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_HUFFMAN_DC_DECODE = [STAGE_RLE, STAGE_DPCM, STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_RLE_DECODE = [STAGE_DPCM, STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_DPCM_DECODE = [STAGE_ZIGZAG, STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_DEZIGZAG = [STAGE_QUANT, STAGE_DCT, STAGE_MCUS]
    SKIP_DEQUANTIZATION = [STAGE_DCT, STAGE_MCUS]
    SKIP_IDCT = [STAGE_MCUS]

    # Variables to track Huffman-encoded data
    encoded_dc_y = encoded_dc_cb = encoded_dc_cr = None
    encoded_ac_y = encoded_ac_cb = encoded_ac_cr = None

    # Reverse Step 11: Parse JPEG bitstream
    if last_encoding_stage not in SKIP_JPEG_PARSING:
        if encoding_result.jpeg_bitstream is None:
            raise ValueError("JPEG stage requires jpeg_bitstream in encoding_result")

        logger.info("Reverse Step 11: Parsing JPEG bitstream...")
        huffman_bitstream, quant_lum, quant_chrom = parse_jpeg_bitstream(
            encoding_result.jpeg_bitstream
        )

        encoding_result.quantization_table_lum = quant_lum
        encoding_result.quantization_table_chrom = quant_chrom

    # Reverse Step 10: Deinterleave (separate Huffman-encoded DC and AC for each MCU)
    if last_encoding_stage not in SKIP_DEINTERLEAVE:
        if huffman_bitstream is None:
            if encoding_result.huffman_bitstream is None:
                raise ValueError(f"{last_encoding_stage} stage requires huffman_bitstream in encoding_result")
            huffman_bitstream = encoding_result.huffman_bitstream

        logger.info("Reverse Step 10: Deinterleaving bitstream...")
        encoded_dc_y, encoded_dc_cb, encoded_dc_cr, encoded_ac_y, encoded_ac_cb, encoded_ac_cr = deinterleave(
            huffman_bitstream,
            num_mcus
        )
    else:
        # Load from encoding result if deinterleaving was skipped
        if encoding_result.encoded_dc_y is not None:
            encoded_dc_y = encoding_result.encoded_dc_y
            encoded_dc_cb = encoding_result.encoded_dc_cb
            encoded_dc_cr = encoding_result.encoded_dc_cr
        if encoding_result.encoded_ac_y is not None:
            encoded_ac_y = encoding_result.encoded_ac_y
            encoded_ac_cb = encoding_result.encoded_ac_cb
            encoded_ac_cr = encoding_result.encoded_ac_cr

    # Reverse Step 9: Huffman decode AC coefficients
    if last_encoding_stage not in SKIP_HUFFMAN_AC_DECODE:
        if encoded_ac_y is None:
            raise ValueError("AC decoding requires encoded_ac_* data")

        logger.info("Reverse Step 9: Huffman decoding AC coefficients...")
        rle_y = huffman_decode_ac(encoded_ac_y, huffman_tables.AC_Y)
        rle_cb = huffman_decode_ac(encoded_ac_cb, huffman_tables.AC_CbCr)
        rle_cr = huffman_decode_ac(encoded_ac_cr, huffman_tables.AC_CbCr)
    else:
        # Load from encoding result
        if encoding_result.rle_y is not None:
            rle_y = encoding_result.rle_y
            rle_cb = encoding_result.rle_cb
            rle_cr = encoding_result.rle_cr

    # Reverse Step 8: Huffman decode DC coefficients
    if last_encoding_stage not in SKIP_HUFFMAN_DC_DECODE:
        if encoded_dc_y is None:
            raise ValueError("DC decoding requires encoded_dc_* data")

        logger.info("Reverse Step 8: Huffman decoding DC coefficients...")
        dpcm_y = huffman_decode_dc(encoded_dc_y, huffman_tables.DC_Y)
        dpcm_cb = huffman_decode_dc(encoded_dc_cb, huffman_tables.DC_CbCr)
        dpcm_cr = huffman_decode_dc(encoded_dc_cr, huffman_tables.DC_CbCr)
    else:
        # Load from encoding result
        if encoding_result.dpcm_y is not None:
            dpcm_y = encoding_result.dpcm_y
            dpcm_cb = encoding_result.dpcm_cb
            dpcm_cr = encoding_result.dpcm_cr

    # Reverse Step 7: RLE decode AC coefficients
    if last_encoding_stage not in SKIP_RLE_DECODE:
        if rle_y is None:
            raise ValueError("RLE decoding requires rle_* data")

        logger.info("Reverse Step 7: Decoding RLE for AC coefficients...")
        ac_y = rle_decode_mcus(rle_y)
        ac_cb = rle_decode_mcus(rle_cb)
        ac_cr = rle_decode_mcus(rle_cr)
    else:
        # Load from encoding result
        if encoding_result.ac_y is not None:
            ac_y = encoding_result.ac_y
            ac_cb = encoding_result.ac_cb
            ac_cr = encoding_result.ac_cr

    # Reverse Step 6: DPCM decode DC coefficients
    if last_encoding_stage not in SKIP_DPCM_DECODE:
        if dpcm_y is None:
            raise ValueError("DPCM decoding requires dpcm_* data")

        logger.info("Reverse Step 6: Decoding DPCM for DC coefficients...")
        dc_y = dpcm_decode(dpcm_y)
        dc_cb = dpcm_decode(dpcm_cb)
        dc_cr = dpcm_decode(dpcm_cr)
    else:
        # Load from encoding result
        if encoding_result.dc_y is not None:
            dc_y = encoding_result.dc_y
            dc_cb = encoding_result.dc_cb
            dc_cr = encoding_result.dc_cr

    # Reverse Step 5: Reconstruct quantized blocks from DC and AC coefficients (dezigzag)
    if last_encoding_stage not in SKIP_DEZIGZAG:
        if dc_y is None or ac_y is None:
            raise ValueError("Dezigzag requires dc_* and ac_* data")

        logger.info("Reverse Step 5: Reconstructing quantized blocks from DC and AC coefficients...")
        num_blocks = len(dc_y)
        zigzag_y = np.empty((num_blocks, 64), dtype=ac_y.dtype)
        zigzag_cb = np.empty((num_blocks, 64), dtype=ac_cb.dtype)
        zigzag_cr = np.empty((num_blocks, 64), dtype=ac_cr.dtype)
        zigzag_y[:, 0] = dc_y
        zigzag_y[:, 1:] = ac_y
        zigzag_cb[:, 0] = dc_cb
        zigzag_cb[:, 1:] = ac_cb
        zigzag_cr[:, 0] = dc_cr
        zigzag_cr[:, 1:] = ac_cr
        quant_y = dezigzag(zigzag_y)
        quant_cb = dezigzag(zigzag_cb)
        quant_cr = dezigzag(zigzag_cr)
    else:
        # Load from encoding result
        if encoding_result.quant_y is not None:
            quant_y = encoding_result.quant_y
            quant_cb = encoding_result.quant_cb
            quant_cr = encoding_result.quant_cr

    # Reverse Step 3: Dequantization
    if last_encoding_stage not in SKIP_DEQUANTIZATION:
        if quant_y is None:
            raise ValueError("Dequantization requires quant_* data")
        if encoding_result.quantization_table_lum is None or encoding_result.quantization_table_chrom is None:
            raise ValueError("Dequantization requires quantization tables in encoding_result")

        logger.info("Reverse Step 3: Applying dequantization...")
        dct_y, dct_cb, dct_cr = dequantize(
            quant_y, quant_cb, quant_cr,
            encoding_result.quantization_table_lum,
            encoding_result.quantization_table_chrom
        )
    else:
        # Load from encoding result
        if encoding_result.dct_y is not None:
            dct_y = encoding_result.dct_y
            dct_cb = encoding_result.dct_cb
            dct_cr = encoding_result.dct_cr

    # Reverse Step 2: IDCT transform
    if last_encoding_stage not in SKIP_IDCT:
        if dct_y is None:
            raise ValueError("IDCT requires dct_* data")

        logger.info("Reverse Step 2: Applying inverse DCT...")
        num_mcus = len(dct_y)
        mcus_dct = np.array([[dct_y[i], dct_cb[i], dct_cr[i]] for i in range(num_mcus)])
        mcus_idct = IDCT(mcus_dct)

        mcus_y = mcus_idct[:, 0] + 128
        mcus_cb = mcus_idct[:, 1] + 128
        mcus_cr = mcus_idct[:, 2] + 128

        detect_errors(mcus_y, mcus_cb, mcus_cr)
    else:
        # Load from encoding result
        if encoding_result.mcus_y is not None:
            mcus_y = encoding_result.mcus_y
            mcus_cb = encoding_result.mcus_cb
            mcus_cr = encoding_result.mcus_cr

    # Reverse Step 1: Reconstruct image from MCUs
    if mcus_y is None:
        raise ValueError("MCU reconstruction requires mcus_* data")

    logger.info("Reverse Step 1: Reconstructing image from MCUs...")
    ycbcr_array = mcus_to_ycbcr_array(mcus_y, mcus_cb, mcus_cr, img_width, img_height)

    return ycbcr_array

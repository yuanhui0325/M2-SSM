"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np
from .scan_writer import build_scan_bytes_444
from util import print_3x3_mcus, EncodingResult, logger
from util.quantization_tables import quantization_table_lum, quantization_table_chrom
from util import huffman_tables
from util.encoding_stages import (
    STAGE_JPEG, STAGE_INTERLEAVER, STAGE_AC, STAGE_DC,
    STAGE_RLE, STAGE_DPCM, STAGE_ZIGZAG, STAGE_QUANT,
    STAGE_DCT, STAGE_MCUS
)
from .partitioning import partition
from .transform import transform
from .quantization import quantize
from .zigzag import zigzag
from .dpcm import dpcm_encode
from .run_length_encoding import rle_encode_mcus
from .bitstream_builder import build_bitstream


def encode(
    y_channel: np.ndarray,
    cb_channel: np.ndarray,
    cr_channel: np.ndarray,
    img_width: int,
    img_height: int,
    last_encoding_stage: str = STAGE_JPEG,
    verbose: bool = False
) -> EncodingResult:
    """Run JPEG encoding pipeline up to specified stage."""
    logger.info("Starting JPEG encoding pipeline")
    result = EncodingResult(img_width=img_width, img_height=img_height)

    # Step 1: Partition into MCUs
    logger.info("Partitioning channels into MCUs...")
    result.mcus_y = partition(y_channel)
    result.mcus_cb = partition(cb_channel)
    result.mcus_cr = partition(cr_channel)
    if verbose:
        print_3x3_mcus(result.mcus_y, result.mcus_cb, result.mcus_cr, "creating MCUs")

    if last_encoding_stage == STAGE_MCUS:
        return result

    # Step 2: DCT
    logger.info("Applying DCT...")
    result.dct_y = transform(result.mcus_y)
    result.dct_cb = transform(result.mcus_cb)
    result.dct_cr = transform(result.mcus_cr)
    if verbose:
        print_3x3_mcus(result.dct_y, result.dct_cb, result.dct_cr, "DCT")

    if last_encoding_stage == STAGE_DCT:
        return result

    # Step 3: Quantization
    logger.info("Applying quantization...")
    result.quant_y = quantize(result.dct_y, quantization_table_lum)
    result.quant_cb = quantize(result.dct_cb, quantization_table_chrom)
    result.quant_cr = quantize(result.dct_cr, quantization_table_chrom)
    result.quantization_table_lum = quantization_table_lum
    result.quantization_table_chrom = quantization_table_chrom
    if verbose:
        print_3x3_mcus(result.quant_y, result.quant_cb, result.quant_cr, "Quantization")

    if last_encoding_stage == STAGE_QUANT:
        return result

    # Step 4: Zigzag ordering
    logger.info("Applying zigzag ordering...")
    zigzag_y = zigzag(result.quant_y)
    zigzag_cb = zigzag(result.quant_cb)
    zigzag_cr = zigzag(result.quant_cr)

    # Step 5: Split DC and AC
    logger.info("Splitting DC and AC coefficients...")
    result.dc_y = zigzag_y[:, 0]
    result.dc_cb = zigzag_cb[:, 0]
    result.dc_cr = zigzag_cr[:, 0]
    result.ac_y = zigzag_y[:, 1:]
    result.ac_cb = zigzag_cb[:, 1:]
    result.ac_cr = zigzag_cr[:, 1:]

    if last_encoding_stage == STAGE_ZIGZAG:
        return result

    # Step 6: DPCM encoding for DC
    logger.info("Applying DPCM to DC coefficients...")
    result.dpcm_y = dpcm_encode(result.dc_y)
    result.dpcm_cb = dpcm_encode(result.dc_cb)
    result.dpcm_cr = dpcm_encode(result.dc_cr)

    if last_encoding_stage == STAGE_DPCM:
        return result

    # Step 7: RLE encoding for AC
    logger.info("Applying RLE to AC coefficients...")
    result.rle_y = rle_encode_mcus(result.ac_y)
    result.rle_cb = rle_encode_mcus(result.ac_cb)
    result.rle_cr = rle_encode_mcus(result.ac_cr)

    if last_encoding_stage == STAGE_RLE:
        return result

    # Step 8-10: Huffman encode + interleave
    logger.info("Huffman encoding and interleaving...")
    huff_tables = {
        "DC_Y": huffman_tables.DC_Y,
        "AC_Y": huffman_tables.AC_Y,
        "DC_CbCr": huffman_tables.DC_CbCr,
        "AC_CbCr": huffman_tables.AC_CbCr
    }
    result.huffman_scan_bytes = build_scan_bytes_444(
        result.dpcm_y, result.rle_y,
        result.dpcm_cb, result.rle_cb,
        result.dpcm_cr, result.rle_cr,
        huff_tables
    )

    # Step 11: Build bitstream
    logger.info("Building JPEG bitstream...")
    result.jpeg_bitstream = build_bitstream(
        quantization_table_lum,
        quantization_table_chrom,
        img_height,
        img_width,
        huff_tables,
        result.huffman_scan_bytes
    )

    logger.info("JPEG encoding completed successfully!")
    return result

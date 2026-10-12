"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""

# Encoding stage constants (in order of encoding pipeline)
STAGE_MCUS = "MCUs"               # After MCU partitioning
STAGE_DCT = "DCT"                 # After DCT transform
STAGE_QUANT = "quant"             # After quantization
STAGE_ZIGZAG = "zigzag"           # After zigzag ordering and DC/AC split
STAGE_DPCM = "DPCM"               # After DPCM encoding DC coefficients
STAGE_RLE = "RLE"                 # After RLE encoding AC coefficients
STAGE_DC = "DC"                   # After Huffman encoding DC coefficients
STAGE_AC = "AC"                   # After Huffman encoding AC coefficients
STAGE_INTERLEAVER = "Interleaver"  # Interleaved Huffman-encoded bitstream
STAGE_JPEG = "jpeg"              # Complete JPEG file

# All valid stages (in reverse order for CLI display - most complete first)
ALL_STAGES = [
    STAGE_JPEG,
    STAGE_INTERLEAVER,
    STAGE_AC,
    STAGE_DC,
    STAGE_RLE,
    STAGE_DPCM,
    STAGE_ZIGZAG,
    STAGE_QUANT,
    STAGE_DCT,
    STAGE_MCUS,
]

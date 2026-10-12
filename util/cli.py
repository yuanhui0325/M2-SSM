"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import argparse
from .encoding_stages import ALL_STAGES, STAGE_JPEG


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="JPEG Encoder - Compress images using JPEG compression algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="test-img/monkey.tiff",
        help="Input image file path"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="out.jpg",
        help="Output JPEG file path"
    )

    parser.add_argument(
        "-d", "--decode-stage",
        type=str,
        choices=ALL_STAGES,
        default=STAGE_JPEG,
        dest="last_encoding_stage",
        help="Encoding stage at which to stop and decode for visualization"
    )

    parser.add_argument(
        "-r", "--reconstructed",
        type=str,
        default="rec.png",
        help="Output file path for reconstructed/decoded image"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Skip decoding/visualization step"
    )

    return parser.parse_args()

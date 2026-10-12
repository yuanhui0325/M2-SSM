"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from pathlib import Path

import numpy as np
import cv2
from logging import DEBUG

# Import application modules
from util import parse_arguments, ycbcr_to_rgb, rgb_to_ycbcr, logger
from util.write_bitstream import write_bitstream_to_file
from encoder import encode
from decoder import decode


def main() -> int:
    """
    Main entry point for the JPEG encoder application.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(DEBUG)
        for handler in logger.handlers:
            handler.setLevel(DEBUG)

    if not Path(args.input).is_file():
        logger.error(f"Input file '{args.input}' does not exist.")
        return 1

    # Read image using OpenCV (loads as BGR)
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        logger.error(f"Failed to load image: {args.input}")
        return 1

    # Convert BGR to RGB
    pixel_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = pixel_array.shape

    logger.debug(f"Image {args.input} loaded - Width: {img_width}px, Height: {img_height}px")

    # Convert from RGB to YCbCr
    ycbcr_array = rgb_to_ycbcr(pixel_array)

    # Split into separate Y, Cb, Cr channels
    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]

    # Run the encoding pipeline
    encoding_result = encode(
        y_channel,
        cb_channel,
        cr_channel,
        img_width,
        img_height,
        args.last_encoding_stage,
        args.verbose)

    # Write JPEG file if we have a complete bitstream
    if encoding_result.jpeg_bitstream is not None:
        write_bitstream_to_file(encoding_result.jpeg_bitstream, args.output)

    # If decoding is enabled, decode and save the image
    if not args.no_decode:
        # Decode to YCbCr array
        ycbcr_array = decode(encoding_result, args.last_encoding_stage)

        # Convert YCbCr to RGB
        decoded_image_rgb = ycbcr_to_rgb(ycbcr_array)

        # Save reconstructed image using OpenCV
        # OpenCV uses BGR, so convert RGB to BGR before saving
        decoded_image_bgr = cv2.cvtColor(decoded_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.reconstructed, decoded_image_bgr)
        logger.info(f"Decoded image saved as {args.reconstructed}")

    return 0


if __name__ == "__main__":
    main()

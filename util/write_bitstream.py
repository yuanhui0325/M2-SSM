"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from util.logger import logger


def write_bitstream_to_file(bytestream: bytes, filename: str) -> None:
    """
    Write JPEG bitstream to file.

    Args:
        bytestream: Complete JPEG bitstream
        filename: Output filename
    """
    with open(filename, "wb") as binary_file:
        bytes_written = binary_file.write(bytestream)
        if bytes_written <= 1000:
            logger.info(f"Written {bytes_written} Bytes to {filename}")
        else:
            logger.info(f"Written {bytes_written/1000} KBytes to {filename}")

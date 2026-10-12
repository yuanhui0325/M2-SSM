"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from bitstring import BitArray


def bits_required(n: int) -> int:
    """
    Calculate number of bits required to store a value.

    Args:
        n: The value to encode

    Returns:
        Number of bits required
    """
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def value_to_bits(value: int) -> BitArray:
    """
    Convert a value to its bit representation.

    Args:
        value: The value to convert

    Returns:
        BitArray representation of the value
    """
    result = BitArray()
    if value > 0:
        result.append("uint:" + str(bits_required(value)) + "=" + str(value))
    elif value < 0:
        value = abs(value)
        result.append("uint:" + str(bits_required(value)) + "=" + str(value))
        result.invert()
    return result


def fill_up_last_byte(bit_string: BitArray) -> BitArray:
    """
    Fill up the last byte with zeros if needed.

    Since the scan data must end at a byte boundary, there may be
    some leftover bits that are simply filled with zeros.

    Args:
        bit_string: BitArray to fill up

    Returns:
        BitArray with filled last byte
    """
    bits_last_byte = len(bit_string) % 8
    if bits_last_byte != 0:
        bit_string.append(BitArray(8 - bits_last_byte))
    return bit_string


def add_FF00(bit_string: BitArray) -> bytearray:
    """
    Add 0x00 after every 0xFF in the bitstream.

    If 'FF' is found in the image data '00' needs to be added afterwards
    as it could be mistaken for a marker otherwise.

    Args:
        bit_string: BitArray containing image data

    Returns:
        bytearray with FF00 stuffing applied
    """
    bytes = bytearray(bit_string.bytes)
    index = 0
    for byte in bytes:
        if byte == int(0xFF):
            bytes.insert(index + 1, int(0x00))
        index += 1
    return bytes

"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""


class BitWriter:
    def __init__(self):
        self._out = bytearray()
        self._cur = 0          # current byte being filled
        self._nbits = 0        # number of bits currently in _cur (0..7)

    def write(self, value: int, nbits: int) -> None:
        """
        Write nbits of value to the stream, MSB-first (JPEG requirement).
        Example: write(0b101, 3) writes bits '1','0','1' in that order.
        """
        if nbits <= 0:
            return

        # write from most-significant to least-significant bit
        for i in range(nbits - 1, -1, -1):
            bit = (value >> i) & 1
            self._cur = (self._cur << 1) | bit
            self._nbits += 1

            if self._nbits == 8:
                self._out.append(self._cur & 0xFF)
                self._cur = 0
                self._nbits = 0

    def flush_one(self) -> None:
        """
        JPEG fill-bits: pad with 1s up to next byte boundary.
        """
        if self._nbits == 0:
            return
        while self._nbits != 0:
            self.write(1, 1)

    def flush_zero(self) -> None:
        """
        (Not JPEG-ideal, but kept for compatibility.)
        Pads with 0s up to next byte boundary.
        """
        if self._nbits == 0:
            return
        while self._nbits != 0:
            self.write(0, 1)

    def get_bytes(self) -> bytes:
        if self._nbits != 0:
            # safer: JPEG expects fill bits = 1
            self.flush_one()
        return bytes(self._out)

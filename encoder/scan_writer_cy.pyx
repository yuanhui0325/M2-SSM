# encoder/scan_writer_cy.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cdef class BitWriterCy:
    cdef bytearray out
    cdef unsigned long long buf
    cdef int nbits

    def __cinit__(self):
        self.out = bytearray()
        self.buf = 0
        self.nbits = 0

    cdef inline void write_bits(self, unsigned int code, int clen):
        self.buf = (self.buf << clen) | code
        self.nbits += clen
        while self.nbits >= 8:
            self.nbits -= 8
            b = (self.buf >> self.nbits) & 0xFF
            self.out.append(b)
            # Byte stuffing
            if b == 0xFF:
                self.out.append(0x00)

    cdef inline void flush_one(self):
        cdef int r = self.nbits & 7
        if r:
            # pad with 1s to next byte boundary
            self.write_bits((1 << (8 - r)) - 1, 8 - r)

    def get_bytes(self):
        return bytes(self.out)

cdef inline int mag_size(int v):
    cdef int a = v if v >= 0 else -v
    cdef int s = 0
    while a:
        a >>= 1
        s += 1
    return s

cdef inline unsigned int neg_ampl(int v, int size):
    # (2^size - 1 + v), v is negative
    return ((1 << size) - 1 + v)

cdef inline void build_codes_lens(dict huff_tbl, list codes, list lens):
    cdef int sym
    cdef object bitstr
    for sym, bitstr in huff_tbl.items():
        if bitstr:
            codes[sym] = int(bitstr, 2)
            lens[sym]  = len(bitstr)
        else:
            codes[sym] = 0
            lens[sym]  = 0

def build_scan_bytes_444(
    dpcm_y, rle_y,
    dpcm_cb, rle_cb,
    dpcm_cr, rle_cr,
    huff_tables
):
    """
    Same signature as Python build_scan_bytes_444(..., huff_tables)
    Returns already byte-stuffed scan bytes.
    """
    cdef BitWriterCy bw = BitWriterCy()
    cdef Py_ssize_t n = len(dpcm_y)
    cdef Py_ssize_t i
    cdef int diff, zr, v, size, sym

    # Build (code,lens) arrays once per call
    cdef list dc_y_codes = [0]*256
    cdef list dc_y_lens  = [0]*256
    cdef list ac_y_codes = [0]*256
    cdef list ac_y_lens  = [0]*256
    cdef list dc_c_codes = [0]*256
    cdef list dc_c_lens  = [0]*256
    cdef list ac_c_codes = [0]*256
    cdef list ac_c_lens  = [0]*256

    build_codes_lens(huff_tables["DC_Y"],     dc_y_codes, dc_y_lens)
    build_codes_lens(huff_tables["AC_Y"],     ac_y_codes, ac_y_lens)
    build_codes_lens(huff_tables["DC_CbCr"],  dc_c_codes, dc_c_lens)
    build_codes_lens(huff_tables["AC_CbCr"],  ac_c_codes, ac_c_lens)

    for i in range(n):
        # ---- Y DC ----
        diff = dpcm_y[i]
        size = mag_size(diff)
        bw.write_bits(dc_y_codes[size], dc_y_lens[size])
        if size:
            bw.write_bits(diff if diff > 0 else neg_ampl(diff, size), size)

        # ---- Y AC ----
        for zr, v in rle_y[i]:
            if v == 0:
                sym = zr << 4
                bw.write_bits(ac_y_codes[sym], ac_y_lens[sym])
            else:
                size = mag_size(v)
                sym = (zr << 4) | size
                bw.write_bits(ac_y_codes[sym], ac_y_lens[sym])
                bw.write_bits(v if v > 0 else neg_ampl(v, size), size)

        # ---- Cb DC ----
        diff = dpcm_cb[i]
        size = mag_size(diff)
        bw.write_bits(dc_c_codes[size], dc_c_lens[size])
        if size:
            bw.write_bits(diff if diff > 0 else neg_ampl(diff, size), size)

        # ---- Cb AC ----
        for zr, v in rle_cb[i]:
            if v == 0:
                sym = zr << 4
                bw.write_bits(ac_c_codes[sym], ac_c_lens[sym])
            else:
                size = mag_size(v)
                sym = (zr << 4) | size
                bw.write_bits(ac_c_codes[sym], ac_c_lens[sym])
                bw.write_bits(v if v > 0 else neg_ampl(v, size), size)

        # ---- Cr DC ----
        diff = dpcm_cr[i]
        size = mag_size(diff)
        bw.write_bits(dc_c_codes[size], dc_c_lens[size])
        if size:
            bw.write_bits(diff if diff > 0 else neg_ampl(diff, size), size)

        # ---- Cr AC ----
        for zr, v in rle_cr[i]:
            if v == 0:
                sym = zr << 4
                bw.write_bits(ac_c_codes[sym], ac_c_lens[sym])
            else:
                size = mag_size(v)
                sym = (zr << 4) | size
                bw.write_bits(ac_c_codes[sym], ac_c_lens[sym])
                bw.write_bits(v if v > 0 else neg_ampl(v, size), size)

    bw.flush_one()
    return bw.get_bytes()

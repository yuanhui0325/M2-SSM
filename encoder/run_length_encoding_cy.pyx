# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython-optimized RLE (Run-Length Encoding) for JPEG AC coefficients
Matches run_length_encoding.py logic but with Cython optimizations
"""

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int32_t INT32


def rle_encode_mcus(ac_coefficients_array):
    """
    Cython-optimized RLE encoder for AC coefficients
    Takes array of shape (n_mcus, 63) and returns list of RLE-encoded MCUs
    Each MCU is list of (zero_run, ac_value) tuples
    """
    
    cdef np.ndarray[INT32, ndim=2] ac = np.asarray(ac_coefficients_array, dtype=np.int32)
    
    # Ensure integer dtype
    if ac.dtype.kind not in ("i", "u"):
        ac = ac.astype(np.int32, copy=False)

    cdef int n_mcus = ac.shape[0]
    result = [None] * n_mcus  # Schneller als append

    cdef int m, last_nz, z, coeff
    cdef list row, out

    for m in range(n_mcus):
        # 1) Einmal pro MCU nach Python-ints konvertieren
        row = ac[m].tolist()   # Länge 63

        # 2) last_nz schneller: while rückwärts
        last_nz = 62
        while last_nz >= 0 and row[last_nz] == 0:
            last_nz -= 1

        if last_nz < 0:
            result[m] = [(0, 0)]  # komplett null -> nur EOB
            continue

        out = []
        z = 0

        # 3) Nur bis last_nz iterieren
        for coeff in row[:last_nz + 1]:
            if coeff == 0:
                z += 1
                if z == 16:
                    out.append((15, 0))  # ZRL
                    z = 0
            else:
                out.append((z, coeff))
                z = 0

        # 4) EOB wenn trailing zeros existieren
        if last_nz < 62:
            out.append((0, 0))

        result[m] = out

    return result

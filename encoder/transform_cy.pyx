# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Cython-optimized DCT (Discrete Cosine Transform)
Matches transform.py exactly - same logic, same output
"""

import numpy as np
cimport numpy as np
cimport cython

import ctypes
from pathlib import Path
import subprocess
import sys

ctypedef np.float64_t FLOAT64
ctypedef np.ndarray ndarray

# Load C library (same as Python version)
def _build_dct_library():
    """Loads libdct8x8 for Windows (.dll) or Linux/Mac (.so)."""
    here = Path(__file__).resolve().parent
    c_path = here / "dct8x8.c"

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        so_path = here / "libdct8x8.so"
        if (not so_path.exists()) or (c_path.stat().st_mtime > so_path.stat().st_mtime):
            cmd = [
                "gcc", "-O3", "-fPIC", "-shared",
                "-o", str(so_path),
                str(c_path),
                "-lm",
            ]
            print("Baue C-DCT-Bibliothek:", " ".join(cmd))
            subprocess.check_call(cmd)
        return so_path
    elif sys.platform == "win32":
        dll_path = here / "libdct8x8.dll"
        if dll_path.exists():
            return dll_path
        else:
            return None  # Return None for fallback
    else:
        raise OSError(f"Plattform {sys.platform} nicht unterstützt")

lib_path = _build_dct_library()
_lib = None
_use_c_lib = False

if lib_path is not None:
    try:
        _lib = ctypes.CDLL(str(lib_path))
        _lib.dct8x8.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _lib.dct8x8.restype = None
        _lib.dct8x8_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        _lib.dct8x8_batch.restype = None
        _use_c_lib = True
        print(f"[OK] C-Modul geladen: {lib_path.name}")
    except Exception as e:
        _lib = None
        print(f"[!] C-Modul-Fehler, nutze scipy: {e}")


def transform(ndarray MCU_list):
    """Apply DCT transform - matches original transform.py exactly"""
    cdef int n, i
    cdef ndarray mcu_flat, shifted, out
    cdef ndarray block, dct_block
    
    if _use_c_lib:
        # 1) Reshape to (n*64) if input is (n, 8, 8)
        
        if MCU_list.ndim == 3:
            n = MCU_list.shape[0]
            mcu_flat = MCU_list.reshape(n, 64)
        else:
            mcu_flat = MCU_list
            n = MCU_list.shape[0]
        
        # 2) Convert to float64 and shift by 128
        shifted = mcu_flat.astype(np.float64, copy=False) - 128.0
        shifted = np.ascontiguousarray(shifted)
        
        out = np.empty_like(shifted, dtype=np.float64)
        
        # 3) Call C library with batch processing
        shifted_ptr = shifted.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        _lib.dct8x8_batch(shifted_ptr, out_ptr, n)
        
        # Reshape back to input shape
        if MCU_list.ndim == 3:
            return out.reshape([MCU_list.shape[i] for i in range(MCU_list.ndim)])
        else:
            return out
    else:
        # Fallback: scipy DCT - simple loop (matches original logic)
        try:
            from scipy.fftpack import dct
            
            # Handle both (n, 64) and (n, 8, 8) shapes
            if MCU_list.ndim == 3:
                n = MCU_list.shape[0]
            else:
                n = MCU_list.shape[0]
            
            shifted = MCU_list.astype(np.float64, copy=False) - 128.0
            out = np.empty_like(shifted, dtype=np.float64)
            
            # Process each 8x8 block
            for i in range(n):
                if MCU_list.ndim == 3:
                    block = shifted[i]  # Already (8, 8)
                else:
                    block = shifted[i].reshape(8, 8)
                
                # 2D DCT: apply 1D DCT to rows then columns
                dct_block = dct(dct(block.T, axis=0, norm='ortho').T, axis=0, norm='ortho')
                
                if MCU_list.ndim == 3:
                    out[i] = dct_block
                else:
                    out[i] = dct_block.ravel()
            
            return out
        except ImportError:
            # Pure NumPy fallback
            shifted = MCU_list.astype(np.float64, copy=False) - 128.0
            return shifted

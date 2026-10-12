"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import numpy as np
import subprocess
from pathlib import Path
from distutils.ccompiler import new_compiler


def build_c_library():
    """Compile dct8x8.c to shared library (libdct8x8.dll/.so)"""
    here = Path(__file__).resolve().parent
    c_file = here / "encoder" / "dct8x8.c"

    if not c_file.exists():
        print(f"Warning: {c_file} not found, skipping C library build")
        return

    # ---------------------------------------------------------
    # PART 1: Windows Build (Using MSVC via distutils)
    # ---------------------------------------------------------
    if sys.platform == "win32":
        dll_file = here / "encoder" / "libdct8x8.dll"
        print(f"Building C library: {c_file.name} -> libdct8x8.dll")

        try:
            # Attempts to use the installed Visual Studio compiler (MSVC)
            compiler = new_compiler()
            objects = compiler.compile(
                [str(c_file)],
                output_dir=str(here / "build"),
                extra_preargs=["/O2"],  # /O2 = Maximize Speed optimization
            )
            compiler.link_shared_lib(
                objects,
                "libdct8x8",
                output_dir=str(here / "encoder"),
                extra_preargs=["/DLL"],
            )
            print(f"[OK] C library built successfully: {dll_file.name}")
        except Exception as e:
            # WHY THIS TRY-BLOCK? 
            # If the user doesn't have a C compiler installed, we don't want the setup to crash.
            # The main code is designed to fall back to 'scipy.fftpack.dct' (slower but works)
            # if this C library fails to build.
            print(f"[!] Warning: Could not compile C library: {e}")
            print("    Will fall back to scipy DCT during runtime")

    # ---------------------------------------------------------
    # PART 2: Linux & macOS Build (Using GCC/Clang)
    # ---------------------------------------------------------
    elif sys.platform.startswith("linux") or sys.platform == "darwin":
        so_file = here / "encoder" / "libdct8x8.so"
        print(f"Building C library: {c_file.name} -> libdct8x8.so")

        try:
            # Standard generic command for GCC/Clang
            cmd = [
                "gcc", "-O3", "-fPIC", "-shared", # -O3 = Max optimization, -fPIC = Position Independent Code
                "-o", str(so_file),
                str(c_file),
                "-lm",
            ]
            subprocess.check_call(cmd)
            print(f"[OK] C library built successfully: {so_file.name}")
        except Exception as e:
            # WHY THIS TRY-BLOCK?
            # Similar to Windows: if 'gcc' is missing, catch the error so installation finishes.
            # The project will run in "compatibility mode" (pure Python/Scipy).
            print(f"[!] Warning: Could not compile C library: {e}")
            print("    Will fall back to scipy DCT during runtime")

    else:
        print(f"Warning: Unsupported platform {sys.platform}")


# Build C library before Cython extensions
print("=" * 60)
build_c_library()
print("=" * 60)

# C extension for DCT8x8 library
c_extensions = []

# Cython extensions
cython_extensions = [
    Extension(
        name="encoder.scan_writer_cy",
        sources=["encoder/scan_writer_cy.pyx"],
    ),
    Extension(
        name="encoder.run_length_encoding_cy",
        sources=["encoder/run_length_encoding_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="encoder.transform_cy",
        sources=["encoder/transform_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="encoder.encode_ac_cy",
        sources=["encoder/encode_ac_cy.pyx"],
    ),
    Extension(
        name="decoder.huffman_decode_cy",
        sources=["decoder/huffman_decode.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)

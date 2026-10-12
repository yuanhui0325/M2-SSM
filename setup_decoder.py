"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
from setuptools import setup, Extension
from Cython.Build import cythonize


# Cython extensions for DECODER only
cython_extensions = [
    Extension(
        name="decoder.huffman_decode_cy",
        sources=["decoder/huffman_decode.pyx"],
        # Note: Setuptools/Cython automatically detects the correct compiler 
        # (MSVC on Windows, GCC/Clang on Linux/Mac) so no manual C-build step is needed here.
    ),
]

setup(
    name="jpeg-decoder",
    version="1.0",
    description="JPEG Decoder (Cython-optimized)",
    # The 'setup' function handles cross-platform build logic automatically for extensions
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

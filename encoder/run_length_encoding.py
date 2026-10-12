"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import numpy as np
from typing import List, Tuple

# Use Cython optimized version
from . import run_length_encoding_cy
rle_encode_mcus = run_length_encoding_cy.rle_encode_mcus

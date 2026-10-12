"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
# Use Cython version for AC encoding
from . import encode_ac_cy
encode_ac_coefficients = encode_ac_cy.encode_ac_coefficients

"""
Package initializer for `src.models`.

Imports and exposes the PyTorch based CNN model (Chambon et al. (2018) variant)
for sleep staging tasks.

##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
"""
from .sleep_stager import (
    SleepStager
)


__all__ = [
    # from sleep_stager.py
    "SleepStager"
]
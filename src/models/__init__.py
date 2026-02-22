"""
Docstring for src.models

This module serves as a package initializer for the `src.models` package.
It imports and exposes Pytorch based CNN model (described in Chambon at al. (2018))
for sleep staging tasks.

##### Author: Kartik M. Jalal
##### Last Updated: 02-22-2026
"""

from .sleep_stager_chambon_2018 import (
    SleepStagerChambon2018
)

__all__ = [
    # from sleep_stager_chambon_2018.py
    "SleepStagerChambon2018"
]
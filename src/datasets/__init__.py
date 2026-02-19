"""
Docstring for src.datasets

This module serves as a package initializer for the `src.datasets` package.
It imports and exposes dataset classes that are used across the project.

~~ Author: Kartik M. Jalal
~~ Last Updated: 02-19-2026
"""

from .epochs import (
    EpochsDataset
)


__all__ = [
    # from epochs.py
    "EpochsDataset"
]
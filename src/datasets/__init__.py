"""
Docstring for src.datasets

This module serves as a package initializer for the `src.datasets` package.
It imports and exposes dataset class, split function that are used across the project.

##### Author: Kartik M. Jalal
##### Last Updated: 02-22-2026
"""

from .epochs import (
    EpochsDataset
)

from .split import (
    split_by_subject
)

__all__ = [
    # from epochs.py
    "EpochsDataset",
    # from split.py
    "split_by_subject"
]
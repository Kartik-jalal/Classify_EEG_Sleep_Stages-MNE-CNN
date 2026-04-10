"""
Package initializer for `src.datasets`.

Exposes the EpochsDataset class for wrapping EEG/EOG epoch arrays as a 
PyTorch Dataset, and the split_by_subject function for subject-wise 
train/val/test splitting.

##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
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
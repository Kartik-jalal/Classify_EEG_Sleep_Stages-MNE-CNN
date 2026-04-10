"""
PyTorch Dataset wrapper for pre-extracted EEG and EOG epoch arrays.

This module defines the EpochsDataset class, which wraps numpy arrays of epoch
data and labels into a PyTorch Dataset that can be fed into a DataLoader for
training deep learning models on sleep staging tasks.

##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
"""
import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Callable


class EpochsDataset(Dataset):
    """
    Expose pre-extracted EEG/EOG epoch arrays as a PyTorch Dataset.

    Parameters
    ----------
    - epochs_data : np.ndarray
        The epochs data, shape (n_epochs, n_channels, n_times).
    - epochs_labels : np.ndarray
        The epochs labels, shape (n_epochs,).
    - subject_id : int | None
        The subject ID for this dataset, if available.
    - recording_id : int | None
        The recording ID for this dataset, if available.
    - transform : Callable | None
        A function applied to each epoch for preprocessing
        (e.g., scaling). Defaults to None.
    """

    def __init__(
        self,
        epochs_data: np.ndarray,
        epochs_labels: np.ndarray,
        subject_id: int | None = None,
        recording_id: int | None = None,
        transform: Callable | None = None
    ):
        assert len(epochs_data) == len(epochs_labels), "Data and labels must have the same number of epochs"

        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.subject_id = subject_id
        self.recording_id = recording_id
        self.transform = transform

    def __len__(self):
        return len(self.epochs_data)
    
    def __getitem__(self, index):
        # get the epoch data and label for the given index
        X = self.epochs_data[index] # shape (n_channels, n_times)
        y = self.epochs_labels[index] # sleep stage label for this epoch

        # apply the transform function to the epoch data (per-sample standardization)
        # EEG and EOG amplitudes vary wildly between subjects and even within a
        # night (e.g., due to electrode impedance changes or sweat). Global scaling
        # (calculating mean/std over the entire dataset) would fail to correct for
        # these local shifts. Therefore, local scaling forces the network to look at
        # the relative shape/pattern of the waves (morphology) rather than the
        # absolute voltage amplitude, which is exactly what we want for sleep
        # staging. Therefore, local scaling is standard and often preferred.
        if self.transform:
            X = self.transform(X)

        # Convert to PyTorch tensors
        # Add a "depth" dimension for Conv2d input
        # (n_channels, n_times) -> (1, n_channels, n_times)
        # where n_channels = 3 (2 EEG + 1 EOG) and n_times = 3000 (30s @ 100 Hz)
        # Note: We use torch.float32 explicitly as it's the standard for DL models
        # and ensures compatibility with GPU computations. Using the default float
        # type (float64) can lead to increased memory usage and slower computations
        # without any benefit for model performance.
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        # For labels, we convert to long (int64) type which is required for PyTorch's 
        # and standard for classification loss functions.
        y = torch.tensor(y, dtype=torch.long)

        return X, y


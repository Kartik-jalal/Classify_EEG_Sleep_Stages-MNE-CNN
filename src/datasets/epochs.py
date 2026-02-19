"""
Docstring for src.datasets.__init__.py.epochs

This module defines the EpochsDataset class, which is a PyTorch Dataset wrapper for
MNE Epochs objects. It allows us to easily convert MNE Epochs into a format that can be
fed into a PyTorch DataLoader for training deep learning models on sleep staging tasks.

~~ Author: Kartik M. Jalal
~~ Last Updated: 02-19-2026
"""
import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Callable

class EpochsDataset(Dataset):
    """
    Class to expose an MNE Epoch object as PyTorch Dataset.

    Parameters:
    epochs_data : np.ndarray
        The epochs data, shape (n_epochs, n_channels, n_times).
    epochs_labels : np.ndarray
        The epochs labels, shape (n_epochs,).
    subject_id : int | None
        The subject ID for this dataset, if available.
    recording_id : int | None
        The recording ID for this dataset, if available.
    transform : Callable | None
        The funcation is eventually apply to each epoch for preprocessing 
        (e.g., scaling). Default to None.
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
        # if provided. For EEG/sleep staging, local scaling is standard and often
        # preferred. 
        # EEG amplitudes vary wildly between subjects and even within a night 
        # (e.g., due to electrode impedance changes or sweat). Global scaling 
        # (calculating mean/std over the entire dataset) would fail to correct for
        # these local shifts. Therefore, local scaling forces the network to look at
        # the relative shape/pattern of the waves (morphology) rather than the 
        # absolute voltage amplitude, which is exactly what we want for sleep 
        # staging.
        if self.transform:
            X = self.transform(X)

        # Convert to PyTorch tensors
        # Add a "channel" dimension for CNN input 
        # (n_channels, n_times) -> (1, n_channels, n_times)
        # Note: We use torch.float32 explicitly as it's the standard for DL models
        # and ensures compatibility with GPU computations. Using the default float 
        # type (float64) can lead to increased memory usage and slower computations
        # without any benefit for model performance.
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        # For labels, we convert to long (int64) type which is required for PyTorch's and
        # standard for classification loss functions.
        y = torch.tensor(y, dtype=torch.long)

        return X, y


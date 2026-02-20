"""
Docstring for src.utils.__init__.py.preprocessing

This module contains functions for preprocessing the raw sleep staging data, such as
epoch extraction and standardization needed for training a sleep stage classifier.

##### Author: Kartik M. Jalal
##### Last Updated: 02-19-2026
"""

import mne
import numpy as np

def extract_epochs(raw: mne.io.Raw, epoch_length: float = 30.0) -> mne.Epochs:
    """
    Extract non overlapping epochs from the raw data.

    Parameters:
    - raw (mne.io.Raw): The raw EEG data.
    - epoch_length (float): The length of each epoch in seconds. Default is 30 sec
        because standard sleep staging (AASM rules) uses 30-second epochs..

    Returns:
    - mne.Epochs: The extracted epochs.
    """
    # Map annotations to event IDs
    # Note: We merge S3 and S4 into the same ID (4) for modern N3 staging.
    event_ids = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4, 
        'Sleep stage R': 5
    }

    # converting the annotations to events
    # This automatically splits long annotations into chunk_duration chunks. This 
    # is vital for sleep data. Sleep annotations usually span long periods 
    # (e.g., "Sleep stage 2" for 10 minutes). chunk_duration tells MNE to chop those
    #  long annotations into smaller, consecutive events of 
    # chunk_duration/epoch_length (30s).
    # Example: A 60s "Sleep stage 2" annotation becomes two separate events of 30s
    # each, both with ID 3.
    events, _ = mne.events_from_annotations(
        raw=raw,
        event_id=event_ids,
        chunk_duration=epoch_length
    ) 


    # define the final event_id mapping for the epochs. This provides readable 
    # labels for your plots and analysis
    event_ids = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4, # merged S3 and S4
        'Sleep stage R': 5
    }

    # --- Creating the Epochs object ---
    # defining the epochs time window. Since, tmax is inclusive for Epochs in MNE,
    # so we subtract one sample duration to get exact length. For example, if sampling
    # rate is 100Hz, one sample is 0.01s and if you set tmax=30.0, you get samples 
    # from 0.0 to 30.0. That's 30.01 seconds of data (one extra sample)!
    # By subtracting one sample duration, you get exactly 0.0 to 29.99 seconds, which
    # is exactly 30 seconds of data (3000 samples). This prevents off-by-one errors
    # when stacking arrays later.
    tmax = epoch_length - (1 / raw.info['sfreq'])

    # create the epochs object
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_ids,
        tmin=0.0, # Start of each epoch relative to the event time.
        tmax=tmax, # end of each epoch
        baseline=None, # We do NOT want to apply baseline correction (subtracting the mean of a pre-stimulus period) because sleep stages don't have a "pre-stimulus" baseline. We want the raw signal amplitude.
        preload=True, # load all epochs into memory for faster processing (if you have enough RAM)
        on_missing='ignore' # Safely handle cases where an event ID might be missing
    )

    return epochs


def scale_epoch(X: np.ndarray) -> np.ndarray:
    """
    Standard scaling of data along the time dimension (axis 1).
    
    Ensures each channel in the epoch has mean 0 and std 1.

    Parameters:
    - X (np.ndarray): The input epoch data, shape (n_channels, n_times).

    Returns:
    - np.ndarray: The scaled epoch data, same shape as input.
    """

    # Calculate mean and std along the time dimension (axis 1) for each channel
    mean = np.mean(X, axis=1, keepdims=True) # shape (n_channels, 1)
    std = np.std(X, axis=1, keepdims=True) # shape (n_channels, 1)

    # Avoid division by zero by adding a small epsilon to std
    epsilon = 1e-10
    std += epsilon

    # Standardize the data: (X - mean) / std
    X_scaled = (X - mean) / std

    return X_scaled
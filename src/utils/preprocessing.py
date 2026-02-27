"""
Docstring for src.utils.__init__.py.preprocessing

This module contains functions for preprocessing the raw sleep staging data, such as
filtering and saving the raw EEG data, epoch extraction and standardization, and creating
epochs dataset needed for training a sleep stage classifier.

##### Author: Kartik M. Jalal
##### Last Updated: 02-27-2026
"""

import mne
import numpy as np
import pathlib

from typing import Any

from src.datasets import EpochsDataset


def filter_and_save_raw_data(
    raw : mne.io.Raw, 
    l_freq : Any, 
    h_freq : Any, 
    output_dir : pathlib.PosixPath
) -> pathlib.PosixPath:
    """
    Loads, filters and save the raw EEG data.

    Parameters
    ----------
    - raw : mne.io.Raw
        The raw EEG data recording to perform the filtering on.
    - l_freq : Any
        Low-frequency cut-off; frequencies below this value will be filtered out.
    - h_freq : Any
        High-frequency cut-off; frequencies above this value will be filtered out.
    - output_dir : pathlib.PosixPath
        Directory where the filtered EEG data will be saved.

    Returns
    -------
    - save_path : pathlib.PosixPath
        Path to the filtered EEG data saved file.

    """
    # Load data into memory before filtering (required by MNE)
    raw.load_data()

    # perform band-pass filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    # save the filtered raw data
    save_path = output_dir / f"{raw.filenames[0].stem}_filtered_raw.fif",
    raw.save(
        fname=save_path[0],
        overwrite=True
    )

    return save_path[0]


def extract_epochs(raw: mne.io.Raw, epoch_length: float = 30.0) -> mne.Epochs:
    """
    Extract non overlapping epochs from the raw data.

    Parameters
    ----------
    - raw (mne.io.Raw): The raw EEG data.
    - epoch_length (float): The length of each epoch in seconds. Default is 30 sec
        because standard sleep staging (AASM rules) uses 30-second epochs..

    Returns
    -------
    - mne.Epochs: The extracted epochs.
    """
    # Map annotations to event IDs
    # Note: We merge S3 and S4 into the same ID (4) for modern N3 staging.
    event_ids = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3, 
        'Sleep stage R': 4
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
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3/4': 3, # merged S3 and S4
        'Sleep stage R': 4
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

    Parameters
    ----------
    - X (np.ndarray): The input epoch data, shape (n_channels, n_times).

    Returns
    -------
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


def create_epochs_ds(
    raw : mne.io.Raw,
    eeg_epoch_duration : float
) -> EpochsDataset:
    """
    Extracts epochs of a single recording given the epochs window duration, and using
    them create the EpochsDataset object.

    Parameters
    ----------
    - raw : mne.io.Raw
        The raw EEG data recording
    - eeg_epoch_duration : float
        The epochs window duration
    
    Returns
    -------
    - epoch_ds : EpochsDataset
        An epoch dataset object of type `torch.utils.data.Dataset`.
    """
    # get the epochs objects
    epochs = extract_epochs(raw=raw, epoch_length=eeg_epoch_duration)

    # extract the data arrats needed for the Dataset class
    # Note: We copy them to ensure they are standard NumPy arrays (not memory maps).
    X = epochs.get_data(copy=True)  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1] # shape: (n_epochs,), The last/third column contains the event IDs
    
    # get metadata
    subject_id = raw.info["subject_info"]["id"]
    recording_id = int(raw.info["subject_info"]["his_id"].split(" ")[-1])

    # create the dataset
    epoch_ds = EpochsDataset(
        epochs_data=X,
        epochs_labels=y,
        subject_id=subject_id,
        recording_id=recording_id,
        transform=scale_epoch
    )

    return epoch_ds
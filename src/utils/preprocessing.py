"""
Preprocessing utilities for the sleep staging pipeline.

This module contains functions for preprocessing the raw digital signal data
(EEG + EOG), including band-pass filtering, ICA-based artefact removal, epoch
extraction and standardization, and creating the EpochsDataset needed for
training a sleep stage classifier.

##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
"""
import mne
import numpy as np
import pathlib
import gc

from typing import Any

from src.datasets import EpochsDataset


def filter_and_save_raw_data(
    raw : mne.io.Raw,
    l_freq : float,
    h_freq : float,
    output_dir : pathlib.Path
) -> pathlib.Path:
    """
    Band-pass filter and save the raw digital signal data (EEG + EOG channels).

    Parameters
    ----------
    - raw : mne.io.Raw
        The raw digital signal data recording (EEG and EOG channels) to
        perform the filtering on.
    - l_freq : float
        Low-frequency cut-off; frequencies below this value will be filtered out.
    - h_freq : float
        High-frequency cut-off; frequencies above this value will be filtered out.
    - output_dir : pathlib.Path
        Directory where the filtered digital signal data will be saved.

    Returns
    -------
    - save_path : pathlib.Path
        Path to the filtered digital signal data saved file.
    """
    # Load data into memory before filtering (required by MNE)
    raw.load_data()

    # perform band-pass filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    # save the filtered raw data
    save_path = output_dir / f"{raw.filenames[0].stem}_filtered_raw.fif"
    raw.save(
        fname=save_path,
        overwrite=True
    )

    return save_path


def perform_ica(
    raw_pair : tuple,
    output_dir : pathlib.Path
) -> pathlib.Path :
    """
    Performs Independent Component Analysis (ICA) to remove EOG (eye movement)
    artefacts from the raw EEG data.

    ICA decomposes the multi-channel EEG signal into statistically independent
    source components. Some of these components capture brain activity while
    others capture artefacts (e.g., eye blinks, eye movements). By identifying
    and excluding the artefact components, we can reconstruct a cleaner EEG
    signal without destroying the underlying brain oscillations — unlike simple
    frequency filtering, which would remove entire frequency bands (including
    Delta waves critical for N3 deep sleep detection).

    Why two separate Raw objects?
    ----------------------------
    ICA source separation works best when fit on data with a wider frequency
    range (1–40 Hz), because eye artefacts span roughly 0–15 Hz and the extra
    high-frequency content helps Picard distinguish brain vs. artefact sources.
    However, for downstream sleep staging we want the final cleaned signal
    band-passed at 0.5–30 Hz (preserving Delta while removing muscle/line
    noise). So we:
      1. Fit ICA on the *original* raw data filtered to 1–40 Hz (wider band).
      2. Apply the learned unmixing weights to the *already filtered* 0.5–30 Hz
         data, removing only the identified artefact components.

    Parameters
    ----------
    - raw_pair : tuple
        A tuple of (old_raw, filtered_raw) where:
        - old_raw : mne.io.Raw
            The original unfiltered recording. Will be filtered in-place to 1–40 Hz 
            for ICA fitting and then discarded.
        - filtered_raw : mne.io.Raw
            The recording already band-pass filtered to 0.5–30 Hz. ICA artefact removal
            will be applied to this copy.
    - output_dir : pathlib.Path
        Directory where the ICA-cleaned, filtered data will be saved as a `.fif` file.

    Returns
    -------
    - save_path : pathlib.Path
        Path to the saved ICA-cleaned and filtered `.fif` file.
    """
    # Unpack the old (unfiltered) and filtered raw data
    old_raw, filtered_raw = raw_pair

    # ----- Step 1: Filter the original raw data with a wider band (1–40 Hz) -----
    # A higher low-cut (1 Hz vs 0.5 Hz) removes slow drifts that can destabilise
    # ICA convergence, while the higher high-cut (40 Hz vs 30 Hz) retains more
    # spectral information for better source separation. This wider-band copy is
    # used *only* for fitting ICA — it is discarded afterwards.
    old_raw.load_data().filter(l_freq=1, h_freq=40)


    # ----- Step 2: Initialise the ICA object -----
    # n_components=2:
    #   With only 2 EEG channels (Fpz-Cz and Pz-Oz) the maximum number of
    #   independent components we can extract is 2. This means ICA performs a
    #   pure unmixing (rotation) without dimensionality reduction — one
    #   component will ideally capture mostly brain activity and the other
    #   mostly artefact. With more channels you could set this lower for
    #   PCA-based dimensionality reduction before ICA.
    #
    # method='picard':
    #   Picard (Preconditioned ICA for Real Data) is a modern ICA solver that
    #   is faster and more numerically stable than the classic FastICA or
    #   Infomax algorithms. It uses an L-BFGS-based optimisation that converges
    #   in fewer iterations, especially on short or noisy EEG recordings.
    #
    # max_iter=500:
    #   Maximum number of Picard optimisation iterations. The default (200) can
    #   be insufficient for noisy recordings; 500 gives the solver room to
    #   converge without excessive compute.
    #
    # fit_params=dict(ortho=True, extended=True):
    #   - ortho=True: Enforces orthogonal (uncorrelated) components, which
    #     improves numerical stability for low-channel-count data like ours.
    #   - extended=True: Uses the extended Picard formulation that can handle
    #     both sub-Gaussian and super-Gaussian source distributions, making it
    #     more robust to the mix of brain and artefact signals.
    #
    # random_state=42:
    #   Fixes the random seed for reproducible decompositions across runs.
    ica = mne.preprocessing.ICA(
        n_components=2,
        method='picard',
        max_iter=500,
        fit_params=dict(ortho=True, extended=True),
        random_state=42
    )


    # ----- Step 3: Fit ICA on the wider-band data -----
    # We fit only on EEG channels (picks='eeg') so the EOG channel is not
    # included as an ICA source — it is reserved as a reference for artefact
    # detection in Step 4. After fitting, old_raw is no longer needed.
    ica.fit(old_raw, picks='eeg')
    del old_raw


    # ----- Step 4: Detect EOG-correlated components -----
    # First, create short epochs time-locked to detected eye blinks/movements
    # in the EOG channel. MNE's create_eog_epochs automatically finds EOG
    # events using peak detection on the EOG channel.
    #   - reject=None: Accept all epochs regardless of amplitude (we trust
    #     the data after filtering).
    #   - baseline=(None, -0.2): Baseline-correct each EOG epoch using the
    #     window from epoch start (tmin=-0.5s) to -0.2s. This removes any DC
    #     offset so the blink/movement waveform stands out clearly.
    #   - tmin=-0.5, tmax=0.5: Extract 1-second windows centred around each
    #     detected EOG event, giving enough context to capture the full blink
    #     waveform.
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw=filtered_raw,
        reject=None,
        baseline=(None, -0.2),
        tmin=-0.5,
        tmax=0.5
    )

    # find_bads_eog correlates each ICA component's time course with the EOG
    # channel across the blink epochs. Components with high correlation are
    # flagged as artefactual. Returns a list of component indices to exclude.
    eog_bad_ic_inds, _ = ica.find_bads_eog(
        eog_epochs
    )

    # Mark the identified artefact components for exclusion. When ICA is
    # applied in Step 5, these components will be zeroed out during the
    # inverse transform, effectively removing their contribution from the
    # reconstructed signal.
    ica.exclude = eog_bad_ic_inds
    del eog_epochs


    # ----- Step 5: Apply ICA to the filtered data -----
    # ica.apply() reconstructs the signal by: (1) projecting the data into
    # component space using the unmixing matrix, (2) zeroing out the excluded
    # components, and (3) projecting back to sensor space. This surgically
    # removes artefact sources while preserving brain oscillations — including
    # Delta waves (0.5–4 Hz) that are essential for detecting N3 deep sleep.
    # The operation modifies filtered_raw in-place.
    # Note: MNE checks which channels the ICA was fitted on and only touches those
    # channels. The EOG channel passes through completely unchanged.
    ica.apply(
        inst=filtered_raw.load_data()
    )
    del ica

    # ----- Step 6: Save the cleaned data and free memory -----
    save_path = output_dir / f"{filtered_raw.filenames[0].stem}_ica_cleaned_filtered_raw.fif"
    filtered_raw.save(
        fname=save_path,
        overwrite=True
    )

    del filtered_raw
    gc.collect()

    return save_path



def extract_epochs(raw: mne.io.Raw, signal_epoch_duration: float = 30.0) -> mne.Epochs:
    """
    Extract non-overlapping epochs from the raw digital signal data (EEG + EOG).

    All channels present in the Raw object (2 EEG + 1 EOG) are included in the
    resulting Epochs. No `picks` filter is applied because the EOG channel
    serves as an additional input feature for the CNN — eye movements carry
    useful information for distinguishing REM sleep from other stages.

    Parameters
    ----------
    - raw : mne.io.Raw
        The raw digital signal data (EEG and EOG channels).
    - signal_epoch_duration : float
        The length of each epoch in seconds. Default is 30 sec because
        standard sleep staging (AASM rules) uses 30-second epochs.

    Returns
    -------
    - mne.Epochs
        The extracted epochs with shape (n_epochs, 3, n_times) where
        the 3 channels are Fpz-Cz (EEG), Pz-Oz (EEG), and EOG horizontal.
    """
    # Map annotations to event IDs
    # Note: We merge S3 and S4 into the same ID (3) for modern N3 staging.
    event_ids = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3, 
        'Sleep stage R': 4
    }

    # Convert the annotations to events
    # This automatically splits long annotations into signal_epoch_duration-sized chunks.
    # This is vital for sleep data. Sleep annotations usually span long periods
    # (e.g., "Sleep stage 2" for 10 minutes). chunk_duration tells MNE to chop those
    # long annotations into smaller, consecutive events of signal_epoch_duration (30s).
    # Example: A 60s "Sleep stage 2" annotation becomes two separate events of 30s
    # each, both with ID 3.
    events, _ = mne.events_from_annotations(
        raw=raw,
        event_id=event_ids,
        chunk_duration=signal_epoch_duration
    ) 


    # Define the final event_id mapping for the epochs. This provides readable
    # labels for plots and analysis
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
    tmax = signal_epoch_duration - (1 / raw.info['sfreq'])

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
    Z-score normalize data along the time dimension (axis 1).

    Ensures each channel (EEG and EOG) in the epoch has mean 0 and std 1.

    Parameters
    ----------
    - X : np.ndarray
        The input epoch data, shape (n_channels, n_times).

    Returns
    -------
    - np.ndarray
        The scaled epoch data, same shape as input.
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
    signal_epoch_duration : float
) -> EpochsDataset:
    """
    Extract epochs from a single recording and wrap them as an EpochsDataset.

    Parameters
    ----------
    - raw : mne.io.Raw
        The raw digital signal data recording (EEG and EOG channels).
    - signal_epoch_duration : float
        The duration of each epoch in seconds (e.g., 30.0 for AASM standard).

    Returns
    -------
    - epoch_ds : EpochsDataset
        An epoch dataset object of type `torch.utils.data.Dataset`.
    """
    # get the epochs objects
    epochs = extract_epochs(raw=raw, signal_epoch_duration=signal_epoch_duration)

    # extract the data arrays needed for the Dataset class
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
"""
Docstring for src.utils.__intit__.py.data_loader

This module is responsible for loading the raw sleep staging data recording from the 
downloaded PhysioNet Sleep dataset files.

## Author: Kartik M. Jalal
## Last Updated: 02-16-2026
"""
import os

import mne
import numpy as np


def load_sleep_physionet_raw_data(
    raw_fname: str, 
    annot_fname: str, 
    crop_wake_mins: int = 30
) -> mne.io.Raw:
    """
    This function load_sleep_physionet_raw is designed to load and preprocess a
    single recording from the Sleep Physionet dataset. It handles file reading,
    channel selection, annotation loading, cropping unnecessary wake periods, and
    renaming channels.

    Parameters:
    - raw_fname: The path to the .edf file containing the raw polysomnography (PSG)
        data (EEG, EOG, EMG, etc.).
    - annot_fname: The path to the corresponding annotation file (hypnogram) which
        contains the sleep stages.
    - crop_wake_mins: The number of minutes of "Wake" (W) stage to keep before the
        first sleep stage and after the last sleep stage. This is important because
        recordings often have long periods of wakefulness at the beginning and end
        which aren't useful for training a sleep classifier.

    Returns:
    - mne.io.Raw: MNE Raw object containing the EEG data and annotations with cropped wake periods.
    """
    # load the raw European Data Format (EDF) data from the .edf file
    raw = mne.io.read_raw_edf(
        input_fname=raw_fname,
        exclude= ( # exclude non-EEG channels from loading
            'EOG horizontal', # EOG - 'eog' in MNE channel types
            'Resp oro-nasal', # respiration - 'misc' in MNE channel types
            'EMG submental', # EMG - 'misc' in MNE channel types
            'Temp rectal', # temperature - 'misc' in MNE channel types
            'Event marker' # event markers - 'misc' in MNE channel types
        )
    )
    # load the sleep stage annotations from the annotation file
    annots = mne.read_annotations(fname=annot_fname)

    # set the loaded annotations to the raw data object
    raw.set_annotations(
        annotations=annots, 
        emit_warning=False #  suppresses warnings if the annotations extend beyond the data duration (which can happen with some datasets)
    )

    # crop the wake periods duration from the beginning and end of the recording
    if crop_wake_mins > 0:
        # Create a boolean list where True means the annotation is a sleep stage
        # (1, 2, 3, 4, R) and False means it's a wake stage (W) or unkown.
        # Note: The Sleep Physionet dataset (which this code is designed for) is 
        # quite old. It was annotated using the older Rechtschaffen & Kales (R&K)
        # rules from 1968, which used:
        #   "Sleep stage 1", 
        #   "Sleep stage 2", 
        #   "Sleep stage 3" ("moderate" deep sleep.),
        #   "Sleep stage 4" ("very deep" deep sleep.), and
        #   "REM sleep" for sleep stages, and
        #   "Wake" for wake stages.
        # The mordern AASM Rules for sleep staging uses 5 stages:
        #   W, N1, N2, N3, and R.
        mask = [
            sleep_stage_annot[-1] # Takes the last character of the annotation string (e.g., "Sleep stage 1" -> "1").
            in ['1', '2', '3', '4', 'R'] 
            for sleep_stage_annot in annots.description                           
        ]
        # Get the indices of all sleep stage annotations.
        sleep_event_inds = np.where(mask)[0]

        # get the onset time of the first sleep stage annotation and subtract the
        # desired wake duration (in seconds) to get the new start time for cropping
        # to keep a buffer of wake time before sleep starts.
        tmin = annots[int(sleep_event_inds[0])]['onset'] - (crop_wake_mins * 60) # convert minutes to seconds
        # get the offset time of the last sleep stage annotation and add the desired
        # wake duration (in seconds) to get the new end time for cropping to keep a
        # buffer of wake time after sleep ends.
        tmax = annots[int(sleep_event_inds[-1])]['onset'] + (crop_wake_mins * 60) # convert minutes to seconds

        # crop the raw data to the new start and end times, effectively removing
        # long wake periods at the beginning and end of the recording while keeping
        # a buffer of wake time around the sleep stages.
        raw.crop(
            tmin=tmin, 
            tmax=tmax
        )

    # rename the EEG channels to a consistent naming convention (e.g., "EEG Fpz-Cz" -> "Fpz-Cz")
    ch_names = {
        i : i.replace("EEG ", "") # remove the "EEG " prefix from channel names
        for i in raw.ch_names
        if "EEG" in i
    }
    mne.rename_channels(
        raw.info, 
        ch_names
    )

    # save the subject and recording ID as metadata in the raw.info object for later use
    basename = os.path.basename(raw_fname) # get the filename from the full path, e.g., SC4001E0-PSG.edf
    subject_id, record_id = int(basename[3:5]), int(basename[5]) # e.g., "00" and 1 from "SC4001..."
    raw.info['subject_info'] = {
        'id': subject_id,
        'rec_id': record_id
    }

    return raw
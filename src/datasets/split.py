"""
Subject-wise dataset splitting for train, validation, and test sets.

This module contains functions for randomly splitting a dataset into Train,
Validation, and Test sets based on subject ID, ensuring no data leakage
between splits.

##### Author: Kartik M. Jalal
##### Last Updated: 04-04-2026
"""
import numpy as np
from torch.utils.data import ConcatDataset

from typing import Tuple


def split_by_subject(
    dataset : ConcatDataset,
    n_test_subjects : int,
    n_val_subjects : int,
    seed : int = 42
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset] :
    """
    Randomly split a dataset into Train, Validation, and Test sets based on subject ID.

    Ensures that all recordings from a specific subject are kept together in
    one split to prevent data leakage.

    Parameters
    ----------
    - dataset : ConcatDataset
        A ConcatDataset of EpochsDataset objects, each with a `subject_id`
        attribute identifying which subject the recording belongs to.
    - n_test_subjects : int
        Number of unique subjects to set aside for test.
    - n_val_subjects : int
        Number of unique subjects to set aside for validation.
    - seed : int
        Random seed for reproducibility.

    Returns
    -------
    - train_ds : ConcatDataset
        Training set (remaining subjects).
    - val_ds : ConcatDataset
        Validation set.
    - test_ds : ConcatDataset
        Test set.
    """
    # get the list of all subject IDs in the dataset
    all_subjects = [
        ds.subject_id for ds in dataset.datasets
    ]
    unique_subjects = np.unique(all_subjects)

    # shuffle the subjects
    rng = np.random.RandomState(seed=seed)
    shuffled_subjects = rng.permutation(unique_subjects)

    # select subjects for test, validation and train
    test_subjects = shuffled_subjects[:n_test_subjects]
    val_subjects = shuffled_subjects[
        n_test_subjects : n_test_subjects + n_val_subjects
    ]
    train_subjects = shuffled_subjects[n_test_subjects + n_val_subjects :]

    # get the dataset.datasets indices (per-recording) for each split
    test_subjects_idx = [
        i for i, s in enumerate(all_subjects) if s in test_subjects
    ]
    val_subjects_idx = [
        i for i, s in enumerate(all_subjects) if s in val_subjects
    ]
    train_subjects_idx = [
        i for i, s in enumerate(all_subjects) if s in train_subjects
    ]

    print(f"Total Unique Subjects: {len(unique_subjects)}")
    print(f"Test Subjects ({len(test_subjects)}):\n\t{test_subjects}")
    print(f"Val Subjects ({len(val_subjects)}):\n\t{val_subjects}")
    print(f"Train Subjects ({len(train_subjects)}):\n\t{train_subjects}")

    # wrap each split's recordings into a new ConcatDataset
    test_ds = ConcatDataset([
        dataset.datasets[i] for i in test_subjects_idx
    ])
    val_ds = ConcatDataset([
        dataset.datasets[i] for i in val_subjects_idx
    ])
    train_ds = ConcatDataset([
        dataset.datasets[i] for i in train_subjects_idx
    ])

    return train_ds, val_ds, test_ds
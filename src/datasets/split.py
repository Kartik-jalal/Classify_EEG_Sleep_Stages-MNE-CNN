"""
Docstring for src.datasets.__init__.py.split

This module contains functions for randomly splits a dataset into Train, Validation, and
Test sets based on SUBJECT ID.

##### Author: Kartik M. Jalal
##### Last Updated: 02-20-2026
"""


import numpy as np
from torch.utils.data import ConcatDataset

from typing import Tuple

def split_by_subject(
    dataset : ConcatDataset,
    n_subjects_test : int,
    n_subjects_val : int,
    seed : int = 42
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset] :
    """
    Randomly splits a dataset into Train, Validation, and Test sets based on SUBJECT ID.
    Ensures that all recordings from a specific subject are kept together in one set.

    Parameters
    ----------
    - dataset : ConcatDataset
        The dataset to split.
    - n_subject_test : int
        Number of unique subjects to set aside for test.
    - n_subject_val : int
        Number of unique subjects to set aside for validation.
    - seed : int
        Random seed for reproducibility.

    Returns
    -------
    - train_ds, val_ds, test_ds : ConcatDataset
        Train, Validation, and Test sets
    """
    # get the list of all subject IDs in the dataset
    all_subjects = [
        ds.subject_id for ds in dataset.datasets
    ]
    unique_subjects = np.unique(all_subjects)

    # shuffle the subjects
    rng = np.random.RandomState(seed=seed)
    shuffled_subjects = rng.permutation(unique_subjects) # random shuffle

    # select subjects for test, validation and train
    test_subjects = shuffled_subjects[:n_subjects_test]
    val_subjects = shuffled_subjects[
        n_subjects_test : n_subjects_test + n_subjects_val
    ]
    train_subjects = shuffled_subjects[n_subjects_test + n_subjects_val :]

    # get the orig indices of the subjects for each split
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

    # create the subsets
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
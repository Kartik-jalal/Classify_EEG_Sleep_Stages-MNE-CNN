"""
A quick training script to train the SleepStager model on the Sleep Physionet dataset
using the preprocessed ICA cleaned data.

This script performs the following steps:
    1. Loads the preprocessed ICA cleaned raw EEG data.
    2. Creates epochs datasets from the raw data.
    3. Splits the dataset into training, validation, and test sets based on subjects.
    4. Computes class weights to handle class imbalance in the training data.
    5. Initializes the SleepStager model and moves it to GPU if available.
    6. Defines DataLoader objects for the training, validation, and test sets.
    7. Sets up the optimizer (AdamW), loss function (CrossEntropyLoss with class weights),
       and learning rate scheduler (ReduceLROnPlateau).
    8. Trains the model while evaluating on the validation set and logging metrics to
       MLflow.
    9. Evaluates the final trained model on the test set and logs the test performance
       metrics to MLflow.

Note: Make sure to have the preprocessed ICA cleaned data ready in the specified 
directory before running this script.

##### Author: Kartik M. Jalal
##### Last Update: 04-10-2026
"""

import mne
import torch
import numpy as np
import matplotlib
import pathlib
from tqdm.contrib.concurrent import thread_map, process_map
from functools import partial
import gc
import yaml
from torch.utils.data import ConcatDataset
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

from src.utils import create_epochs_ds
from src.datasets import split_by_subject
from src.models import SleepStager
from src.utils import fit_model
from src.utils import evaluate

matplotlib.use("QtAgg") # use the Qt backend for interactive plotting
mne.set_log_level("ERROR")  # suppress MNE info and warnings for cleaner output




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Using GPU for computations. Training should be faster.")
    else:
        print(
            "No GPU found. Using CPU for computations, training might be slower."
            "\n\nIf running on Google Colab, make sure to enable GPU acceleration in the notebook settings."
        )

    # ----- Load all the params -----
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    # ----- Loading ICA cleaned Raw data -----
    sleep_physionet_data_ica_dir = pathlib.Path(config["data"]["dir"]["ica_cleaned"]) # path where the filtered EEG data iss saved

    ica_cleaned_filtered_raw_files = []
    # get all the ica cleaned file names 
    for s_p_data_ica_fname in sleep_physionet_data_ica_dir.iterdir():
        ica_cleaned_filtered_raw_files.append(pathlib.Path(s_p_data_ica_fname))

    # load the data
    ica_cleaned_filtered_raws = thread_map(
        mne.io.read_raw_fif,
        ica_cleaned_filtered_raw_files,
        max_workers=2,
        desc="Loading the ICA cleaned filtered raw data (preload=False)"
    )

    # clear the memory 
    del ica_cleaned_filtered_raw_files
    gc.collect()


    # ----- Creating Epochs -----
    # define the partial function with fixed arguments for creating the epochs ds
    partial_fn = partial(
        create_epochs_ds,
        signal_epoch_duration=config["data"]["digital_signal"]["epoch_duration"]
    )

    # store epochs dataset in parallel across multiple CPU cores.
    datasets = process_map(
        partial_fn,
        ica_cleaned_filtered_raws,
        max_workers=4,
        chunksize=1,
        desc="Processing and creating epochs datasets"
    )

    # concatinate all the datasets into one big "datasets"
    dataset = ConcatDataset(datasets)
    print(f"Total number of epochs in the combined dataset: {len(dataset)}")


    # ----- Creating DS -----
    # calculate number of subjects for splits (e.g., 20% Test, 20% Val)
    total_subjects = len(
        np.unique(
            [ds.subject_id for ds in dataset.datasets]
        )
    )
    n_test_subjects = max(1, int(total_subjects * config["data"]["dataset"]["test_subjects_pct"]))
    n_val_subjects = max(1, int(total_subjects * config["data"]["dataset"]["val_subjects_pct"]))

    # perform split
    train_ds, valid_ds, test_ds = split_by_subject(
        dataset=dataset,
        n_test_subjects=n_test_subjects,
        n_val_subjects=n_val_subjects
    )

    # Note: the epochs here means different chucks of that particular dataset / iteration
    print(f"Train size: {len(train_ds)} epochs")
    print(f"Val size: {len(valid_ds)} epochs")
    print(f"Test size: {len(test_ds)} epochs")


    # ----- Compute Class Weight -----
    classes_mapping  = {
        0: "Sleep stage W", 
        1: "Sleep stage 1", 
        2: "Sleep stage 2", 
        3: "Sleep stage 3", 
        4: "Sleep stage R"
    }

    # get all the labels from the training set
    train_y = np.concatenate(
        [ds.epochs_labels for ds in train_ds.datasets]
    )

    # calculate weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_y),
        y=train_y
    )

    # convert to a PyTorch tensor for the Loss Function later
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    for i, class_weight in enumerate(class_weights):
        print(f"{classes_mapping[i]} : {class_weight}")

    
    # ----- Initialise the Model -----
    # sampling rate
    sfreq = ica_cleaned_filtered_raws[0].info['sfreq']
    # number of EEG channels
    n_channels = len(ica_cleaned_filtered_raws[0].ch_names)


    # Intialise the model
    model = SleepStager(
        signal_epoch_duration=config["data"]["digital_signal"]["epoch_duration"],
        n_channels=n_channels,
        sfreq=sfreq,
        n_classes=5,
        n_spatial_filters=config["model"]["architecture"]["n_spatial_filters"],
        n_temporal_filters_l1=config["model"]["architecture"]["n_temporal_filters_l1"],
        n_temporal_filters_l2=config["model"]["architecture"]["n_temporal_filters_l2"],
        temp_conv_size_sec=config["model"]["architecture"]["temp_conv_size_sec"],
        max_pool_size_sec=config["model"]["architecture"]["max_pool_size_sec"],
        dropout_rate=config["model"]["architecture"]["dropout_rate"]
    )

    # move model to GPU if CUDA is avilable for significantly faster training
    print(f"Using device: {device}")
    model = model.to(device)


    # ----- Define DataLoader objects -----
    # Create DataLoaders
    loader_train = DataLoader(
        train_ds,
        batch_size=config["data"]["dataset"]["train_batch_size"],
        shuffle=True,
        num_workers=2
    )
    loader_val = DataLoader(
        valid_ds,
        batch_size=config["data"]["dataset"]["val_batch_size"],
        shuffle=False
    )
    loader_test = DataLoader(
        test_ds,
        batch_size=config["data"]["dataset"]["val_batch_size"],
        shuffle=False
    )


    # ----- Setting-Up Optimiser, Loss Function & LR Schedular
    # loss function 
    criterion = CrossEntropyLoss(
        weight=class_weights_tensor # we pass our previously calculated class_weights to handle the class imbalance
    ).to(device=device) # put the criterion to GPU/CPU

    # optimiser
    # Split parameters into decay and no-decay groups.
    # Weight decay should only apply to Conv2d/Linear weights — not biases or BatchNorm
    # params. BatchNorm's gamma (scale) should stay near 1.0 and beta (shift) should
    # freely learn offsets; decay would fight both by pushing them toward 0.
    decay_params = []
    no_decay_params = []

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            no_decay_params.extend(m.parameters())
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            decay_params.append(m.weight)
            if m.bias is not None:
                no_decay_params.append(m.bias)

    optimiser = AdamW(
        params=[
            {'params': decay_params, 'weight_decay': float(config["model"]["optimiser"]["weight_decay"])},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=float(config["model"]["optimiser"]["lr"]),
    )


    # LR Schedular
    scheduler = ReduceLROnPlateau(
        optimizer=optimiser,
        mode=config["model"]["scheduler"]["mode"], 
        factor=config["model"]["scheduler"]["factor"], 
        patience=config["model"]["scheduler"]["patience"],
        threshold=float(config["model"]["scheduler"]["threshold"])
    )


    # ----- Training, Evaluating and Testing the Model -----
    # point MLflow to mlflow local server and set an experiment name to group your runs
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name=config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=config["mlflow"]["run_name"], nested=True) as run:
        mlflow.log_params(config)

        print(f"Started MLflow Run ID: {run.info.run_id}")
        baseline_training_mlflow_run_id = run.info.run_id
        
        # fit the model - training and evaluating the model
        best_model, history = fit_model(
            model=model,
            loader_train=loader_train,
            loader_valid=loader_val,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
            metric_fns={"cohen_kappa_score" : cohen_kappa_score}, # Using Cohen's Kappa to account for class imbalance
            n_epochs=config["model"]["training"]["n_epochs"],
            patience=config["model"]["training"]["early_stopping_patience"],
            scheduler=scheduler,
            max_grad_norm=config["model"]["training"]["max_grad_norm"]
        )

        # --- Final Evaluation on the Test Set
        # get the Cohen Kappa Score
        test_loss, test_performance = evaluate(
            model=model,
            loader=loader_test,
            criterion=criterion,
            device=device,
            metric_fns={"cohen_kappa_score" : cohen_kappa_score, "balanced_accuracy_score" : balanced_accuracy_score},
            desc="Testing"
        )

        mlflow.log_metrics({
            "test_loss" : test_loss,
            "test_cohen_kappa_score" : test_performance["cohen_kappa_score"],
            "test_balanced_accuracy_score" : test_performance["balanced_accuracy_score"]
        })

        print(f"\nFinal Test Set Performance:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Cohen's kappa score: {test_performance["cohen_kappa_score"]:.4f}")
        print(f"Test Balanced accuracy score: {test_performance["balanced_accuracy_score"]:.4f}")



if __name__ == "__main__":
    main()
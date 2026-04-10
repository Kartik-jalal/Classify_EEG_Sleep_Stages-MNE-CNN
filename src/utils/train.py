"""
Training, evaluation, and model fitting utilities for sleep staging.

This module contains the training loop, evaluation, and fit orchestration
functions for the sleep staging CNN, including early stopping, learning rate
scheduling, and optional MLflow metric logging.

##### Author: Kartik M. Jalal
##### Last Updated: 04-05-2026
"""
import copy
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import mlflow
from tqdm.auto import tqdm

from typing import Callable, Tuple, Union, Optional


def train_one_epoch(
    model : torch.nn.Module,
    loader : torch.utils.data.DataLoader,
    optimiser : torch.optim.Optimizer,
    criterion : torch.nn.Module,
    device : torch.device,
    metric_fns : dict,
    max_grad_norm : float | None
) -> Tuple[float, dict]:
    """
    Execute one complete pass (epoch) over the training dataset.

    Parameters
    ----------
    - model : torch.nn.Module
        The PyTorch model to train.
    - loader : torch.utils.data.DataLoader
        The DataLoader providing training batches.
    - optimiser : torch.optim.Optimizer
        The optimiser (e.g., Adam) used to update model weights.
    - criterion : torch.nn.Module
        The loss function (e.g., CrossEntropyLoss).
    - device : torch.device
        The device (CPU or CUDA GPU) to perform computations on.
    - metric_fns : dict
        A dictionary of functions to calculate performance metrics
        (e.g., Cohen's Kappa Score, Balanced Accuracy Score).
    - max_grad_norm : float | None
        The maximum allowed norm for gradients. If not None, gradients are clipped.

    Returns
    -------
    - avg_loss : float
        The average loss across all batches in this epoch.
    - performance_metrics : dict
        A dictionary containing the performance metrics calculated over the entire epoch.
    """
    # Set model to training mode, which enables Dropout, BatchNorm updates, etc.
    model.train()

    losses = []
    y_true_all, y_pred_all = [], []

    # Wrap the data loader with tqdm
    pbar = tqdm(
        loader, 
        desc="Training", 
        leave=False # leave=False makes the bar disappear after the epoch finishes  so your notebook doesn't get cluttered with 50 completed bars!
    )
    

    # iterate over all batches in the DataLoader
    for batch_x, batch_y in pbar:
        # Move data to the correct device (CPU/GPU)
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.long) # CrossEntropy expects Long tensors

        # clear calculated old gradients from the previous steps
        optimiser.zero_grad(set_to_none=True) # set_to_None=True is slightly faster/more memory efficient than False

        # -- Forward pass
        # compute model predictions
        logits = model(batch_x)
        # compute the loss
        loss = criterion(logits, batch_y)

        # -- Backpropagation
        # Backward Pass : compute gradients
        loss.backward()
        # Gradient Clipping - safety net to prevent exploding gradients
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        # update the weights using the optimiser
        optimiser.step()

        # store this batch loss
        losses.append(
            loss.item() # .item() extracts the scalar value from the 1-element loss tensor
        ) 

        # get the predicted sleep stage index
        preds = torch.argmax(
            logits, dim=1
        ).detach().cpu().numpy() # .detach() stops gradient tracking, .cpu() moves to RAM & .numpy() converts to array.
        # the real labels
        y_true = batch_y.detach().cpu().numpy()
        # store all the predicted and real labels
        y_pred_all.append(preds)
        y_true_all.append(y_true)

        # Update the progress bar text with the live loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # create a one long single list for both
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    # -- compute metrics
    # the average loss over all the batches
    avg_loss = float(np.mean(losses))
    # performance metrics comparing predicted vs real labels
    performance_metrics = {}
    for metric_fn_name, metric_fn in metric_fns.items():
        performance_metrics[metric_fn_name] = metric_fn(y_true, y_pred)

    return avg_loss, performance_metrics


@torch.inference_mode()
def evaluate(
    model : torch.nn.Module,
    loader : torch.utils.data.DataLoader,
    criterion : torch.nn.Module,
    device : torch.device,
    metric_fns : dict,
    desc : str = "Evaluating"
) -> Tuple[float, dict]:
    """
    Evaluate the model on a validation or test dataset.

    Uses @torch.inference_mode() to disable gradient tracking for this function,
    which saves memory and speeds up computations. This is faster than
    torch.no_grad() as it also disables view tracking and version counting.

    Parameters
    ----------
    - model : torch.nn.Module
        The PyTorch model to evaluate/test.
    - loader : torch.utils.data.DataLoader
        The DataLoader providing evaluation/testing batches.
    - criterion : torch.nn.Module
        The loss function (e.g., CrossEntropyLoss).
    - device : torch.device
        The device (CPU or CUDA GPU) to perform computations on.
    - metric_fns : dict
        A dictionary of functions to calculate performance metrics
        (e.g., Cohen's Kappa Score, Balanced Accuracy Score).
    - desc : str | optional
        Description for the tqdm progress bar. Default is "Evaluating".

    Returns
    -------
    - avg_loss : float
        The average loss across all batches.
    - performance_metrics : dict
        A dictionary containing the performance metrics calculated over the entire dataset.
    """
    # set the model to evaluation mode (disables Dropout, freezes BatchNorm)
    model.eval()

    losses = []
    y_true_all, y_pred_all = [], []

    # Wrap evaluation loader.
    pbar = tqdm(
        loader, 
        desc=desc, 
        leave=False
    )
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.long)

        # forward pass only
        logits = model(batch_x)

        # calculate the loss
        loss = criterion(logits, batch_y)
        # store the loss
        losses.append(
            loss.item()
        )

        # get the predicted and real labels
        # Note: .detach() is unnecessary under @torch.inference_mode()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = batch_y.cpu().numpy()
        # store the values
        y_pred_all.append(preds)
        y_true_all.append(y_true)

        # live loss updates
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # combine all into a one long list
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    # calculate the average loss and performance metrics over all batches
    avg_loss = float(np.mean(losses))
    performance_metrics = {}
    for metric_fn_name, metric_fn in metric_fns.items():
        performance_metrics[metric_fn_name] = metric_fn(y_true, y_pred)

    return avg_loss, performance_metrics


def fit_model(
    model : torch.nn.Module,
    loader_train : torch.utils.data.DataLoader,
    loader_valid : torch.utils.data.DataLoader,
    optimiser : torch.optim.Optimizer,
    criterion : torch.nn.Module,
    device : torch.device,
    metric_fns : dict,
    n_epochs : int = 20,
    patience : int = 5,
    scheduler : Optional[Union[LRScheduler, ReduceLROnPlateau]] | None = None,
    max_grad_norm : float = 1.0
) -> Tuple[torch.nn.Module, list]:
    """
    Orchestrate the full training loop with early stopping and metric logging.

    Trains the model for up to n_epochs, evaluates on the validation set each
    epoch, applies learning rate scheduling, and optionally logs all metrics to
    MLflow. Restores the best model weights (by validation loss) before returning.

    Parameters
    ----------
    - model : torch.nn.Module
        The PyTorch model to train/evaluate.
    - loader_train : torch.utils.data.DataLoader
        The DataLoader providing training batches.
    - loader_valid : torch.utils.data.DataLoader
        The DataLoader providing evaluation batches.
    - optimiser : torch.optim.Optimizer
        The optimiser (e.g., Adam) used to update model weights.
    - criterion : torch.nn.Module
        The loss function (e.g., CrossEntropyLoss).
    - device : torch.device
        The device (CPU or CUDA GPU) to perform computations on.
    - metric_fns : dict
        A dictionary of functions to calculate performance metrics
        (e.g., Cohen's Kappa Score, Balanced Accuracy Score).
    - n_epochs : int
        Maximum number of training epochs.
    - patience : int
        Triggers early stopping when validation loss hasn't improved for
        this many consecutive epochs.
    - scheduler : LRScheduler | ReduceLROnPlateau | None
        The learning rate scheduler. If ReduceLROnPlateau, it steps on
        validation loss; otherwise it steps unconditionally each epoch.
    - max_grad_norm : float | None
        The maximum allowed norm for gradients. If not None, gradients are clipped.

    Returns
    -------
    - best_model : torch.nn.Module
        PyTorch model restored to the best validation loss state.
    - history : list
        A list of dicts containing per-epoch training and evaluation logs.
    """
    # initialise the best evaluation avg loss
    best_eval_avg_loss = float("inf")
    # save the initial state in case the model immediately degrades
    best_model_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0
    best_val_epoch = 1
    history = []

    # print info column
    print(
        f"{'Epoch':<6} | "
        f"{'Train Loss':<10} | "
        f"{'Val Loss':<10} | "
        f"{'|'.join([f'{"Train " + metric_fun_name:<10}' for metric_fun_name in metric_fns.keys()])} | "
        f"{'|'.join([f'{"Val " + metric_fun_name:<10}' for metric_fun_name in metric_fns.keys()])} | "
        f"{'Best Val Epoch':<10}"
    )
    print("-" * 99)

    for epoch in range(1, n_epochs+1):
        # train the model
        train_avg_loss, train_metrics = train_one_epoch(
            model=model,
            loader=loader_train,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
            metric_fns=metric_fns,
            max_grad_norm=max_grad_norm
        )

        # evaluate the model
        eval_avg_loss, eval_metrics = evaluate(
            model=model,
            loader=loader_valid,
            criterion=criterion,
            device=device,
            metric_fns=metric_fns
        ) 

        # Step the learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # if/when using ReduceLROnPlateau (in our case) - it needs to look at the
                # validation loss to decide if it should step
                scheduler.step(eval_avg_loss)
            else:
                scheduler.step()


        # log results
        log = {
            "epoch" : epoch,
            "train_loss" : train_avg_loss,
            "evaluation_loss" : eval_avg_loss,
        }
        for train_metric_name, train_metric_score in train_metrics.items():
            log[f"train_{train_metric_name}"] = train_metric_score
        for eval_metric_name, eval_metric_score in eval_metrics.items():
            log[f"eval_{eval_metric_name}"] = eval_metric_score 
        history.append(log)
        

        # Note: When an active run exists globally, mlflow.log_metric automatically attaches to it
        mlflow.log_metric("train_loss", train_avg_loss, step=epoch)
        mlflow.log_metric("evaluation_loss", eval_avg_loss, step=epoch)
        for train_metric_name, train_metric_score in train_metrics.items(): 
            mlflow.log_metric(f"train_{train_metric_name}", train_metric_score, step=epoch)
        for eval_metric_name, eval_metric_score in eval_metrics.items():
            mlflow.log_metric(f"evaluation_{eval_metric_name}", eval_metric_score, step=epoch)


        # early stopping logic
        if eval_avg_loss < best_eval_avg_loss:
            best_eval_avg_loss = eval_avg_loss
            # Deepcopy the weights so we can restore them later
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            best_val_epoch = epoch
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                print(f"Best Validation Loss was: {best_eval_avg_loss:.4f}")
                break

        print( 
            f"{epoch:<6} | "
            f"{train_avg_loss:<10.4f} | "
            f"{eval_avg_loss:<10.4f} | "
            f"{' | '.join([f"{val:<23.4f}" for val in train_metrics.values()])} | "
            f"{' | '.join([f"{val:<21.4f}" for val in eval_metrics.values()])} | "
            f"{best_val_epoch:<10}"
        )

    # restore the best weights before returning the model
    model.load_state_dict(best_model_state)

    return model, history




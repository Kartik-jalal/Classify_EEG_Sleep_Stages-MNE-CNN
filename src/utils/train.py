"""
Docstring for src.utils.__init__.py.train

This module contains functions for training/evaluating/testing a PyTorch model.

##### Author: Kartik M. Jalal
##### Last Updated: 02-27-2026
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
    Executes one complete pass (epoch) over the training dataset.

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
    - metric_fn : dict
        A dictionary, containing functions to calculate performance metrics (e.g., Cohen's Kappa Score, Balance Accuracy Score).
    - max_grad_norm : float | None
        The maximum allowed norm for gradients. If not None, gradients are clipped.

    Returns
    -------
    - avg_loss : float
        The average loss across all batches in this epoch.
    - preform_metrices : dict 
        A dictionary containing the performance metrices calculated over the entire epoch.
    """
    # Set model to training model, which enables Dropout, BatchNorm updates, etc.
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
    # performace metrices compared of the predicted and real labels
    preform_metrices = {}
    for metric_fn_name, metric_fn in metric_fns.items():
        preform_metrices[metric_fn_name] = metric_fn(y_true, y_pred)

    return avg_loss, preform_metrices


@torch.inference_mode()
def evaluate(
    model : torch.nn.Module,
    loader : torch.utils.data.DataLoader,
    criterion : torch.nn.Module,
    device : torch.device,
    metric_fns : dict
) -> Tuple[float, dict]:
    """
    Evaluated the model on a validation or test dataset.

    Uses @torch.inference_mode() to disable gradient calculation globally (given 
    .backward() is not called) for this function, which saves memory and speeds up
    computations during evaluation.
    Even though it is for inference-only, but faster eval-style mode than 
    torch.no_grad() in many cases of validation/testing/inference.
    
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
        A dictionary, containing functions to calculate performance metrics (e.g., Cohen's Kappa Score, Balance Accuracy Score).

    Returns
    -------
    - avg_loss : float
        The average loss across all batches in this epoch.
    - perform_metrices : dict 
        A dictionary, containing the performance metrices calculated over the entire epoch.
    """
    # set the model to evaluation mode (disables Dropout, freezes BatchNorm)
    model.eval()

    losses = []
    y_true_all, y_pred_all = [], []

    # Wrap validation loader.
    pbar = tqdm(
        loader, 
        desc="Validating", 
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
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_true = batch_y.detach().cpu().numpy()
        # store the values
        y_pred_all.append(preds)
        y_true_all.append(y_true)

        # live loss updates
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # combine all into a one long list
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    # calacualte the average loss and performance matric over all the batches
    avg_loss = float(np.mean(losses))
    perform_metrices = {}
    for metric_fn_name, metric_fn in metric_fns.items():
        perform_metrices[metric_fn_name] = metric_fn(y_true, y_pred)

    return avg_loss, perform_metrices


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
    schedular : Optional[Union[LRScheduler, ReduceLROnPlateau]] | None = None,
    max_grad_norm : float = 1.0,
    mlflow_run : bool = False
) -> Tuple[torch.nn.Module, list]:
    """
    The main training loop orchestration function. Handles early stopping, learning
    rate scheduling, and optional MLflow logging.

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
        A dictionary, containing functions to calculate performance metrics (e.g., Cohen's Kappa Score, Balance Accuracy Score).
    - n_epochs : int
        The number of iterations to train the model using loader_train.
    - patience : int
        Triggers early stopping when loss hasn't improved over n (patience) epochs.
    - schedular : Any | None
        The learning rate scheduler. Must have a .step() method.
    - max_grad_norm : float | None
        The maximum allowed norm for gradients. If not None, gradients are clipped.
    - mlflow_run : boolean 
        If True, metrics will be logged to MLflow.
    
    Returns
    -------
    - best_model : torch.nn.Module
        PyTorch model with the best model state.
    - history : list
        A listing containing all epochs training and evaluation logs.
    """
    # initialise the best evaluation avg loss
    best_eval_avg_loss = float("inf")
    # save the initial state in case the model immediately degrades
    best_model_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0
    history = []

    # print info column
    print(
        f"{'Epoch':<6} | "
        f"{'Train Loss':<10} | "
        f"{'Val Loss':<10} | "
        f"{'|'.join([f'{"Train " + metric_fun_name:<10}' for metric_fun_name in metric_fns.keys()])} | "
        f"{'|'.join([f'{"Val " + metric_fun_name:<10}' for metric_fun_name in metric_fns.keys()])}"
    )
    print("-" * 82)

    for epoch in range(1, n_epochs+1):
        # train the model
        train_avg_loss, train_perform_metrices = train_one_epoch(
            model=model,
            loader=loader_train,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
            metric_fns=metric_fns,
            max_grad_norm=max_grad_norm
        )

        # evaluate the model
        eval_avg_loss, eval_perform_metrices = evaluate(
            model=model,
            loader=loader_valid,
            criterion=criterion,
            device=device,
            metric_fns=metric_fns
        ) 

        # Step the Learning Rate Schedular
        if schedular is not None:
            # if/when using ReduceLROnPlateau (in our case) - it needs to look at the
            # validation loss to decide if it should step
            schedular.step(eval_avg_loss)

        # log results
        log = {
            "epoch" : epoch,
            "train_loss" : train_avg_loss,
            "evaluation_loss" : eval_avg_loss,
        }
        for train_perform_metric_name, train_perform_metric_score in train_perform_metrices.items():
            log[f"train_{train_perform_metric_name}"] = train_perform_metric_score
        for eval_perform_metric_name, eval_perform_metric_score in eval_perform_metrices.items():
            log[f"eval_{eval_perform_metric_name}"] = eval_perform_metric_score 
        history.append(log)

        print( 
            f"{epoch:<6} | "
            f"{train_avg_loss:<10.4f} | "
            f"{eval_avg_loss:<10.4f} | "
            f"{' | '.join([f"{val:<23.4f}" for val in train_perform_metrices.values()])} | "
            f"{' | '.join([f"{val:<20.4f}" for val in eval_perform_metrices.values()])}"
        )
        
        # MLflow tracking
        if mlflow_run:
            # Note: When an active run exists globally, mlflow.log_metric automatically attaches to it
            mlflow.log_metric("train_loss", train_avg_loss, step=epoch)
            mlflow.log_metric("evaluation_loss", eval_avg_loss, step=epoch)
            for train_perform_metric_name, train_perform_metric_score in train_perform_metrices.items(): 
                mlflow.log_metric(f"train_{train_perform_metric_name}", train_perform_metric_score, step=epoch)
            for eval_perform_metric_name, eval_perform_metric_score in eval_perform_metrices.items():
                mlflow.log_metric(f"evaluation_{eval_perform_metric_name}", eval_perform_metric_score, step=epoch)


        # early stopping logic
        if eval_avg_loss < best_eval_avg_loss:
            best_eval_avg_loss = eval_avg_loss
            # Deepcopy the weights so we can restore them later
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                print(f"Best Validation Loss was: {best_eval_avg_loss:.4f}")
                break

    # restore the best weights before returning the model
    model.load_state_dict(best_model_state)

    return model, history




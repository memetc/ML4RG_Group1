import json
import argparse
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Dict, Any

from process_data import prepare_data_loaders
from models.simple_cnn import SimpleCNN
from models.cnn_v2 import CNNV2


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0):
        """
        Early stopping to stop the training when the loss does not improve after a given patience.

        Parameters:
        - patience (int): How long to wait after last time the validation loss improved. Default: 5
        - verbose (bool): If True, prints a message for each validation loss improvement. Default: False
        - delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0

        Attributes:
        - patience (int): The patience parameter.
        - verbose (bool): The verbose parameter.
        - delta (float): The delta parameter.
        - counter (int): Counter to keep track of how many epochs have passed without improvement.
        - best_score (float): The best score encountered.
        - early_stop (bool): Flag to indicate whether training should be stopped.
        - val_loss_min (float): Minimum validation loss encountered.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: nn.Module):
        """
        Call method to check if early stopping condition is met.

        Parameters:
        - val_loss (float): The current validation loss.
        - model (nn.Module): The model being trained.

        If the validation loss decreases, the model checkpoint is saved. If the validation loss does not
        improve for a number of epochs equal to the patience parameter, training is stopped.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """
        Saves the model when validation loss decreases.

        Parameters:
            val_loss (float): The current validation loss.
            model (nn.Module): The model being trained.
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), "last_checkpoint.pt")


def initialize_model(config: Dict[str, Any]) -> nn.Module:
    """
    Initializes the model based on the configuration.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: Initialized model.
    """
    if config["model_version"] == 1:
        print("Started training CNNV2")
        return CNNV2(**config)
    else:
        print("Started training SimpleCNN")
        return SimpleCNN(**config)


def train_one_epoch(
    net: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    """
    Trains the model for one epoch.

    Parameters:
        net (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to perform computation on.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.

    Returns:
        float: Training loss for the epoch.
    """
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = [i.to(device) for i in inputs], labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().detach().numpy()
    return running_loss / len(train_loader)


def validate_one_epoch(
    net: nn.Module, val_loader: DataLoader, device: torch.device, criterion: nn.Module
) -> float:
    """
    Validates the model for one epoch.

    Parameters:
        net (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to perform computation on.
        criterion (nn.Module): Loss function.

    Returns:
        float: Validation loss for the epoch.
    """
    net.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = [i.to(device) for i in inputs], labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.cpu().detach().numpy()
            val_steps += 1
    return val_loss / val_steps


def train(
    config: Dict[str, Any],
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Trains the model.

    Parameters:
        config (dict): Configuration dictionary.
        net (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.

    Returns:
        Tuple[nn.Module, list, list]: The trained model, list of training losses, and list of validation losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )
    early_stopping = EarlyStopping(patience=7, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(net, train_loader, device, criterion, optimizer)
        train_losses.append(train_loss)

        val_loss = validate_one_epoch(net, val_loader, device, criterion)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}, train loss: {train_loss}, validation loss: {val_loss}")
        early_stopping(val_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return net, train_losses, val_losses


def predict(
    config: Dict[str, Any], net: nn.Module, test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates predictions using the trained model.

    Parameters:
        config (dict): Configuration dictionary.
        net (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        - all_predictions (np.ndarray): Array of predictions made by the model.
        - all_labels (np.ndarray): Array of true labels from the test dataset.
        - all_species_ids (np.ndarray): Array of species IDs from the test dataset.
        - all_stress_ids (np.ndarray): Array of stress IDs from the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    all_predictions = []
    all_labels = []
    all_species_ids = []
    all_stress_ids = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = [i.to(device) for i in inputs], labels.to(device)
            outputs = net(inputs)
            species_ids = np.argmax(inputs[0].cpu().numpy(), axis=1)
            stress_ids = np.argmax(inputs[1].cpu().numpy(), axis=1)

            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_species_ids.append(species_ids)
            all_stress_ids.append(stress_ids)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_species_ids = np.concatenate(all_species_ids, axis=0)
    all_stress_ids = np.concatenate(all_stress_ids, axis=0)

    return all_predictions, all_labels, all_species_ids, all_stress_ids


def evaluate(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluates the model performance.

    Parameters:
        predictions (np.ndarray): Predictions made by the model.
        labels (np.ndarray): True labels.

    Returns:
        Tuple[float, float, float]: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 score.
    """
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return mse, mae, r2


def random_search(
    train_dataset: pd.DataFrame,
    test_size: float,
    param_dist: Dict[str, Any],
    n_iter: int = 10,
) -> Tuple[Dict[str, Any], nn.Module, DataLoader]:
    """
    Performs a random search for the best hyperparameters.

    Parameters:
        train_dataset (pd.DataFrame): Training dataset.
        test_size (float): Test size ratio.
        param_dist (dict): Dictionary of parameter distributions.
        n_iter (int): Number of iterations for the random search.

    Returns:
        Tuple[dict, nn.Module, DataLoader]: Best configuration, best model, and test data loader.
    """
    best_config = None
    best_val_loss = float("inf")

    for _ in range(n_iter):
        config = {
            "lr": param_dist["lr"].rvs(),
            "cnn_filters": param_dist["cnn_filters"].rvs(),
            "batch_size": param_dist["batch_size"].rvs(),
            "hidden_size": param_dist["hidden_size"].rvs(),
            "activation": random.choice(param_dist["activation"]),
            "epochs": 50,
            "species_id": -1,
            "test_size": test_size,
            "model_version": 1,
        }

        net = initialize_model(config)
        train_loader, val_loader, test_loader = prepare_data_loaders(
            train_dataset, config["batch_size"]
        )

        net, _, val_losses = train(config, net, train_loader, val_loader)

        if min(val_losses) < best_val_loss:
            best_val_loss = min(val_losses)
            best_config = config

        print(f"Val Loss: {min(val_losses)}")

    print(f"Best Config: {best_config}, Best Val Loss: {best_val_loss}")
    return best_config, net, test_loader


def inverse_normalize(values: np.ndarray) -> np.ndarray:
    """
    Inverse normalizes the values by applying the exponential function and subtracting 1.

    Parameters:
        values (np.ndarray): Normalized values.

    Returns:
        np.ndarray: Inverse normalized values.
    """
    return np.exp(values) - 1


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a JSON file.

    Parameters:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def update_config_from_args(config: Dict[str, Any], args: List[str]) -> Dict[str, Any]:
    """
    Updates the configuration dictionary with command line arguments.

    Parameters:
        config (dict): Original configuration dictionary.
        args (list): List of command line arguments.

    Returns:
        dict: Updated configuration dictionary.
    """
    for arg in args:
        key, value = arg.split("=")
        key = key.lstrip("--")
        if key in config:
            if isinstance(config[key], int):
                config[key] = int(value)
            elif isinstance(config[key], float):
                config[key] = float(value)
            else:
                config[key] = value
        else:
            print(f"Warning: {key} not found in the configuration file.")
    return config

def main():
    """
    Main function to run the training and evaluation process.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override configuration options, e.g. batch_size=32",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    config = update_config_from_args(config, args.overrides)

    data_df = pd.read_csv(config["data_path"])
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_df, config["batch_size"]
    )

    net = initialize_model(config)
    net, train_losses, val_losses = train(config, net, train_loader, val_loader)

    predictions, labels, _, _ = predict(config, net, test_loader)
    mse, mae, r2 = evaluate(predictions, labels)

    print(f"Evaluation Results - MSE: {mse}, MAE: {mae}, R2: {r2}")


if __name__ == "__main__":
    main()

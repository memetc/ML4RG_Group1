import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.simple_cnn import SimpleCNN
from models.cnn_v2 import CNNV2
from .helpers import SequenceDataset
from .load_manager import load_data

class EarlyStopping:
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
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        """
        Call method to check if early stopping condition is met.

        Parameters:
        - val_loss (float): The current validation loss.
        - model (torch.nn.Module): The model being trained.

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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), 'last_checkpoint.pt')
        
def train(config):
    """
    Train a neural network model based on the provided configuration.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters for training. 
      It should include the following keys:
        - 'model_version' (int): Version of the model to use (0 for SimpleCNN, 1 for CNNV2).
        - 'lr' (float): Learning rate for the optimizer.
        - 'species_id' (int): ID of the species to filter data by.
        - 'test_size' (int): Number of samples to include in the dataset.
        - 'data_df' (pd.DataFrame): DataFrame containing the dataset.
        - 'batch_size' (int): Batch size for training.
        - 'epochs' (int): Number of epochs for training.

    Returns:
    - net (torch.nn.Module): The trained model.
    - train_losses (list): List of training losses for each epoch.
    - val_losses (list): List of validation losses for each epoch.
    - test_dataset (SequenceDataset): The testing dataset.
    """

    # Initializes the model based on the specified version in the config.
    if config['model_version'] == 1:
        print('Started training CNNV2')
        net = CNNV2(**config)
    elif config['model_version'] == 0:
        print('Started training SimpleCNN')
        net = SimpleCNN(**config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Sets up the device, loss criterion, optimizer, learning rate scheduler, and early stopping.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Loads and preprocesses the training and testing data.
    train_dataset, test_dataset = load_data(species_id=config['species_id'], size=config['test_size'], data_df=config['data_df'])
    print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    train_losses = []
    val_losses = []

    # Trains the model over a specified number of epochs with early stopping.
    for epoch in range(config['epochs']):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = [i.to(device) for i in inputs], labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().detach().numpy()

        train_losses.append(running_loss/len(train_loader))
        
        net.eval()
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = [i.to(device) for i in inputs], labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.cpu().detach().numpy()
            val_steps += 1

        val_losses.append(val_loss/val_steps)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}, train loss: {running_loss/len(train_loader)} validation loss: {val_loss/len(val_loader)}")
        early_stopping(val_loss / val_steps, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # Returns the trained model, training losses, validation losses, and test dataset.        
    return net, train_losses, val_losses, test_dataset


def predict(config, net, test_dataset):
    """
    Generate predictions using the trained model on the test dataset.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters for prediction.
      It should include the following key:
        - 'batch_size' (int): Batch size for the DataLoader.
    - net (torch.nn.Module): The trained model.
    - test_dataset (SequenceDataset): The test dataset.

    Returns:
    - all_predictions (np.ndarray): Array of predictions made by the model.
    - all_labels (np.ndarray): Array of true labels from the test dataset.
    - all_species_ids (np.ndarray): Array of species IDs from the test dataset.
    - all_stress_ids (np.ndarray): Array of stress IDs from the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    all_predictions = []
    all_labels = []
    all_species_ids = []
    all_stress_ids = []
    # Iterates over the test DataLoader to generate predictions
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)

            outputs = net(inputs)
            species_ids = np.argmax(inputs[0].cpu().numpy(), axis=1)
            stress_ids = np.argmax(inputs[1].cpu().numpy(), axis=1)

            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_species_ids.append(species_ids)
            all_stress_ids.append(stress_ids)

    # Collects the predictions, true labels, species IDs, and stress IDs.
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_species_ids = np.concatenate(all_species_ids, axis=0)
    all_stress_ids = np.concatenate(all_stress_ids, axis=0)

    return all_predictions, all_labels, all_species_ids, all_stress_ids
    

def evaluate(predictions, labels):
    """
    Evaluate the model's performance using various metrics.
    
    Parameters:
    - predictions (array-like): The predicted values by the model.
    - labels (array-like): The true values from the dataset.

    Returns:
    - mse (float): The Mean Squared Error of the predictions.
    - mae (float): The Mean Absolute Error of the predictions.
    - r2 (float): The R-squared score of the predictions.
    """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(labels, predictions)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(labels, predictions)

    # Calculate R-squared (R²)
    r2 = r2_score(labels, predictions)
    
    return mse, mae, r2

def random_search(data_df, species_id, test_size, param_dist, n_iter=10):
    """
    Perform a random search to find the best hyperparameters for training a neural network model.

    Parameters:
    - data_df (pd.DataFrame): The DataFrame containing the dataset.
    - species_id (int): The ID of the species to filter data by.
    - test_size (int): The number of samples to include in the test dataset.
    - param_dist (dict): A dictionary containing the distributions of hyperparameters to sample from.
      It should include:
        - 'lr': A distribution for learning rates.
        - 'cnn_filters': A distribution for the number of CNN filters.
        - 'batch_size': A distribution for batch sizes.
        - 'hidden_size': A distribution for hidden layer sizes.
        - 'activation': A list of activation functions to choose from.
    - n_iter (int, optional): The number of iterations to perform. Default is 10.

    Returns:
    - best_config (dict): The best hyperparameter configuration found.
    """
    best_config = None
    best_val_loss = float('inf')

    for i in range(n_iter):
        # Randomly sample hyperparameters
        config = {
            'lr': param_dist['lr'].rvs(),
            'cnn_filters': param_dist['cnn_filters'].rvs(),
            'batch_size': param_dist['batch_size'].rvs(),
            'hidden_size': param_dist['hidden_size'].rvs(),
            'activation': random.choice(param_dist['activation']),
            'epochs': 50,
            'species_id': -1,
            'test_size': test_size,
            'model_version': 1,
            'data_df': data_df
        }

        net, train_losses, val_losses, test_dataset = train(config)
        
        if min(val_losses) < best_val_loss:
            best_val_loss = min(val_losses)
            best_config = config
        
        print(f"Val Loss: {min(val_losses)}")

    print(f"Best Config: {best_config}, Best Val Loss: {best_val_loss}")
    return best_config


def inverse_normalize(values):
    """
    Inverse normalize the values by applying the exponential function and subtracting 1.

    Parameters:
    - values (array-like): The values to be inverse normalized.

    Returns:
    - array-like: The inverse normalized values.
    """
    return np.exp(values) - 1
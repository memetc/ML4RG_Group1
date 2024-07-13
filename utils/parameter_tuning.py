import multiprocess as mp  # Note that we are importing "multiprocess", no "ing"
import random
import torch.nn as nn
import os

from scipy.stats import uniform, randint
from utils.model_utils import train

def evaluate_config(config, train_dataset):
    net, train_losses, val_losses = train(config, train_dataset)
    return min(val_losses), config, net

def generate_config(param_dist, test_size):
    return {
        "lr": param_dist["lr"].rvs(),
        "cnn_filters": param_dist["cnn_filters"].rvs(),
        "batch_size": 1024,
        "hidden_size": param_dist["hidden_size"].rvs(),
        "kernel_size": 6,
        "activation": random.choice(param_dist["activation"]),
        "epochs": 10,
        "species_id": -1,
        "test_size": test_size,
        "model_version": 1,
        "stress_condition_size": 11,
    }

def worker(config, train_dataset, result_queue):
    try:
        print("Worker started to work with", config)
        val_loss, config, net = evaluate_config(config, train_dataset)
        result_queue.put((val_loss, config, net))
    except Exception as e:
        result_queue.put((float("inf"), None, None))
        print(f"An error occurred: {e}")

# def random_search(train_dataset, test_size, param_dist, n_iter=10):
#     """
#     Perform a random search to find the best hyperparameters for training a neural network model.
#
#     Parameters:
#     - train_dataset: The dataset used for training.
#     - test_size (int): The number of samples to include in the test dataset.
#     - param_dist (dict): A dictionary containing the distributions of hyperparameters to sample from.
#       It should include:
#         - 'lr': A distribution for learning rates.
#         - 'cnn_filters': A distribution for the number of CNN filters.
#         - 'batch_size': A distribution for batch sizes.
#         - 'hidden_size': A distribution for hidden layer sizes.
#         - 'activation': A list of activation functions to choose from.
#     - n_iter (int, optional): The number of iterations to perform. Default is 10.
#
#     Returns:
#     - best_config (dict): The best hyperparameter configuration found.
#     - best_net: The best trained network.
#     """
#     print('Random search started')
#     best_config = None
#     best_val_loss = float("inf")
#     best_net = None
#
#     configs = [generate_config(param_dist, test_size) for _ in range(n_iter)]
#     result_queue = mp.Queue()
#     processes = []
#
#     for config in configs:
#         p = mp.Process(target=worker, args=(config, train_dataset, result_queue))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     while not result_queue.empty():
#         val_loss, config, net = result_queue.get()
#         if config is not None and val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_config = config
#             best_net = net
#         print(f"Val Loss: {val_loss}")
#
#     print(f"Best Config: {best_config}, Best Val Loss: {best_val_loss}")
#     return best_config, best_net



def random_search(train_dataset, test_size, param_dist, n_iter=10):
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
    best_val_loss = float("inf")

    for i in range(n_iter):
        # Randomly sample hyperparameters
        config = {
            "lr": param_dist["lr"].rvs(),
            "cnn_filters": int(param_dist["cnn_filters"].rvs()),
            "batch_size": 1024,
            "hidden_size": int(param_dist["hidden_size"].rvs()),
            "kernel_size": 6,
            "activation": random.choice(param_dist["activation"]),
            "epochs": 10,
            "species_id": -1,
            "test_size": test_size,
            "model_version": 1,
            "stress_condition_size": 11,
        }
        print(config)
        net, train_losses, val_losses = train(config, train_dataset)

        if min(val_losses) < best_val_loss:
            best_val_loss = min(val_losses)
            best_config = config
            best_net = net

        print(f"Val Loss: {min(val_losses)}")

    print(f"Best Config: {best_config}, Best Val Loss: {best_val_loss}")
    return best_config, best_net
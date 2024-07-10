import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified split using GroupShuffleSplit to create training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input dataframe with species_name and stress_condition_name columns.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the train_val split to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test dataframes.
    """
    df["group"] = (
        df["species_name"].astype(str) + "_" + df["stress_condition_name"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["group"]))

    df_train_val = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val["group"]))

    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    return df_train, df_val, df_test


def prepare_tensors(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts dataframe columns to PyTorch tensors.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors for sequences, species, stress conditions, and labels.
    """
    sequence_tensor = torch.tensor(
        df["upstream200"].to_list(), dtype=torch.float32
    ).permute(0, 2, 1)
    species_tensor = torch.tensor(df["species"].to_list(), dtype=torch.float32)
    stress_tensor = torch.tensor(df["stress_condition"].to_list(), dtype=torch.float32)
    y_tensor = torch.tensor(df["tpm"].values, dtype=torch.float32).unsqueeze(1)

    return sequence_tensor, species_tensor, stress_tensor, y_tensor


def create_datasets_and_loaders(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch datasets and dataloaders from dataframes.

    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_val (pd.DataFrame): Validation dataframe.
        df_test (pd.DataFrame): Test dataframe.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for training, validation, and test sets.
    """
    sequence_tensor_train, species_tensor_train, stress_tensor_train, y_tensor_train = (
        prepare_tensors(df_train)
    )
    sequence_tensor_val, species_tensor_val, stress_tensor_val, y_tensor_val = (
        prepare_tensors(df_val)
    )
    sequence_tensor_test, species_tensor_test, stress_tensor_test, y_tensor_test = (
        prepare_tensors(df_test)
    )

    train_dataset = TensorDataset(
        sequence_tensor_train, species_tensor_train, stress_tensor_train, y_tensor_train
    )
    val_dataset = TensorDataset(
        sequence_tensor_val, species_tensor_val, stress_tensor_val, y_tensor_val
    )
    test_dataset = TensorDataset(
        sequence_tensor_test, species_tensor_test, stress_tensor_test, y_tensor_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_data_loaders(
    df: pd.DataFrame, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Processes the input dataframe and returns PyTorch dataloaders for training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for training, validation, and test sets.
    """
    df_train, df_val, df_test = stratified_split(df)
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        df_train, df_val, df_test, batch_size
    )
    return train_loader, val_loader, test_loader

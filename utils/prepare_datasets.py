import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from typing import Tuple

NUM_SPECIES = 29
NUM_STESS_CONDITIONS = 9
SEQ_LENGTH = 203


def reshape_sequences(df: pd.DataFrame, embedding_size: int = 4) -> torch.Tensor:
    """
    Reshapes the sequence data from the DataFrame, concatenating one-hot encoded vectors
    for A, T, G, and C horizontally, padding if necessary.

    Args:
    df (pd.DataFrame): DataFrame containing the sequence data.
    column_name (str): The name of the column containing the sequence data.
    embedding_size (int): The embedding size of the sequence data. Default is 4.

    Returns:
    torch.Tensor: The reshaped sequence data as a PyTorch tensor.
    """
    # Extract and stack the sequence data from the DataFrame
    num_batches = len(df)
    sequences = np.vstack(df.loc[:, "upstream200"]).reshape(num_batches, -1)
    seq_length = sequences.shape[1]

    # Calculate padding needed
    padding_needed = (embedding_size - (seq_length % embedding_size)) % embedding_size

    # Pad the sequences if necessary
    if padding_needed > 0:
        padding = np.zeros((num_batches, padding_needed))
        sequences = np.hstack((sequences, padding))

    # Reshape to (batch_size, seq_length, embedding_size)
    sequences = sequences.reshape(num_batches, embedding_size, -1)

    return sequences


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    sample_size: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified split using GroupShuffleSplit to create training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input dataframe with species_name and stress_condition_name columns.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the train_val split to include in the validation split.
        random_state (int): Random state for reproducibility.
        sample_size (float): Fraction (0 < sample_size <= 1) or absolute number (>1) of samples to retain.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test dataframes.
    """
    # Sample the dataframe if sample_size is specified
    if sample_size is not None:
        if sample_size > 1:
            df = df.sample(n=int(sample_size), random_state=random_state)
        else:
            df = df.sample(frac=sample_size, random_state=random_state)

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


def prepare_numpy_arrays(
    df: pd.DataFrame, scaler: StandardScaler = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Converts the input dataframe into numpy arrays for features and target and applies z-normalization to the target.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.
        scaler (StandardScaler): Optional StandardScaler to use for normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]: Arrays for features and normalized target, and the scaler used.
    """
    sequence_array = np.vstack(df.loc[:, "upstream200"]).reshape(len(df), -1)
    species_array = np.vstack(df.loc[:, "species"]).reshape(len(df), -1)
    stress_array = np.vstack(df.loc[:, "stress_condition"]).reshape(len(df), -1)
    y = df["tpm"].to_numpy(dtype=np.float32)

    # if scaler is None:
    #     scaler = StandardScaler()
    #     y = scaler.fit_transform(y)
    # else:
    #     y = scaler.transform(y)

    return sequence_array, species_array, stress_array, y, scaler


def create_datasets_for_xgboost(
    df: pd.DataFrame,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Splits the data into training, validation, and test sets, prepares numpy arrays with scaling,
    and converts the data to DMatrix format for XGBoost.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Numpy arrays for training, validation, and test sets.
    """
    # Split the data
    df_train, df_val, df_test = stratified_split(df)

    # Prepare numpy arrays with scaling
    (
        sequence_array_train,
        species_array_train,
        stress_array_train,
        y_array_train,
        scaler,
    ) = prepare_numpy_arrays(df_train)
    sequence_array_val, species_array_val, stress_array_val, y_array_val, _ = (
        prepare_numpy_arrays(df_val, scaler)
    )
    sequence_array_test, species_array_test, stress_array_test, y_array_test, _ = (
        prepare_numpy_arrays(df_test, scaler)
    )

    X_train = np.concatenate(
        (sequence_array_train, species_array_train, stress_array_train), axis=1
    )
    X_val = np.concatenate(
        (sequence_array_val, species_array_val, stress_array_val), axis=1
    )
    X_test = np.concatenate(
        (sequence_array_test, species_array_test, stress_array_test), axis=1
    )

    return (X_train, y_array_train), (X_val, y_array_val), (X_test, y_array_test)


def prepare_dmatrices(df: pd.DataFrame) -> Tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
    """
    Converts the input dataframe to DMatrix format for XGBoost.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.

    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]: DMatrix objects for training, validation, and test sets.
    """
    (X_train, y_array_train), (X_val, y_array_val), (X_test, y_array_test) = (
        create_datasets_for_xgboost(df)
    )

    dtrain = xgb.DMatrix(X_train, label=y_array_train)
    dval = xgb.DMatrix(X_val, label=y_array_val)
    dtest = xgb.DMatrix(X_test, label=y_array_test)

    return dtrain, dval, dtest


def prepare_tensors(
    df: pd.DataFrame, scaler: StandardScaler = None, embedding_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Converts dataframe columns to PyTorch tensors and applies z-normalization to the target.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.
        scaler (StandardScaler): Optional StandardScaler to use for normalization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]: Tensors for sequences, species, stress conditions, labels, and the scaler used.
    """
    sequence_tensor = torch.tensor(
        reshape_sequences(df, embedding_size), dtype=torch.float32
    )
    species_tensor = torch.tensor(
        np.vstack(df.loc[:, "species"]).reshape(len(df), -1), dtype=torch.float32
    )
    stress_tensor = torch.tensor(
        np.vstack(df.loc[:, "stress_condition"]).reshape(len(df), -1),
        dtype=torch.float32,
    )

    y = np.array(df["tpm"].values, dtype=np.float32).reshape(-1, 1)

    # if scaler is None:
    #     scaler = StandardScaler()
    #     y = scaler.fit_transform(y)
    # else:
    #     y = scaler.transform(y)

    y_tensor = torch.tensor(y, dtype=torch.float32)

    return sequence_tensor, species_tensor, stress_tensor, y_tensor, scaler


def create_datasets_and_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    batch_size: int,
    embedding_size: int = 4,
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
    (
        sequence_tensor_train,
        species_tensor_train,
        stress_tensor_train,
        y_tensor_train,
        scaler,
    ) = prepare_tensors(df_train, embedding_size=embedding_size)
    sequence_tensor_val, species_tensor_val, stress_tensor_val, y_tensor_val, _ = (
        prepare_tensors(df_val, scaler, embedding_size=embedding_size)
    )
    sequence_tensor_test, species_tensor_test, stress_tensor_test, y_tensor_test, _ = (
        prepare_tensors(df_test, scaler, embedding_size=embedding_size)
    )

    train_dataset = TensorDataset(
        torch.concat([sequence_tensor_train, sequence_tensor_val], axis=0),
        torch.concat([species_tensor_train, species_tensor_val], axis=0),
        torch.concat([stress_tensor_train, stress_tensor_val], axis=0),
        torch.concat([y_tensor_train, y_tensor_val], axis=0),
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
    df: pd.DataFrame,
    batch_size: int,
    sample_size: float = None,
    embedding_size: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Processes the input dataframe and returns PyTorch dataloaders for training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input dataframe with necessary columns.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for training, validation, and test sets.
    """
    df_train, df_val, df_test = stratified_split(df, sample_size=sample_size)
    return create_datasets_and_loaders(
        df_train, df_val, df_test, batch_size, embedding_size=embedding_size
    )

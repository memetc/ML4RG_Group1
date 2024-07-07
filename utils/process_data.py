import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from .helpers import SequenceDataset
from .helpers import species_name_to_abb
from .normalizations import ctrl_normalize

import pandas as pd
from typing import Optional

BASES = "ATCGRYSWKMBDHVN"
# The valid characters including IUPAC degenerate base symbols
IUPAC_DEGENERATE_BASES = set(BASES)
# The complement mappings for the DNA bases including IUPAC degenerate base symbols
COMPLEMENT_MAP = str.maketrans(BASES, "TAGCYRSWMKVHDBN")
# One-hot encoding mapping for the DNA bases including IUPAC degenerate base symbols
ONE_HOT_MAP = {base: idx for idx, base in enumerate(BASES)}


def _is_valid_sequence(seq: str) -> bool:
    return len(seq) > 0 and set(seq).issubset(IUPAC_DEGENERATE_BASES)


def complement_dna(sequence: str) -> str:
    """
    Computes the complement of a DNA sequence, including IUPAC degenerate base symbols.

    Args:
        sequence (str): The DNA sequence to complement.

    Returns:
        str: The complement of the DNA sequence.
    """
    return sequence.translate(COMPLEMENT_MAP)


def onehot_encode_dna(sequence: str, max_length: int) -> np.ndarray:
    """
    One-hot encodes a DNA sequence, including IUPAC degenerate base symbols,
    and pads with zeros if the sequence has fewer characters than max_length.

    Args:
        sequence (str): The DNA sequence to one-hot encode.
        max_length (int): The maximum length of the sequence for padding.

    Returns:
        np.ndarray: A one-hot encoded numpy array representation of the DNA sequence.
    """
    # Create a zero matrix of shape (max_length, len(BASES))
    onehot_encoded = np.zeros((max_length, len(BASES)), dtype=int)

    # Fill in the one-hot encoding for each base in the sequence
    for i, base in enumerate(sequence):
        if i >= max_length:
            break
        if base in ONE_HOT_MAP:
            onehot_encoded[i, ONE_HOT_MAP[base]] = 1

    return onehot_encoded


def one_hot_encode_to_numpy(
    df: pd.DataFrame, column_name: str, new_column_name: Optional[str] = None
) -> pd.DataFrame:
    """
    One-hot encodes the specified column of the DataFrame and adds the encoded vectors as numpy arrays in a new column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to be one-hot encoded.
    new_column_name (Optional[str]): The name of the new column to store the one-hot encoded vectors. If None, defaults to '{column_name}_encoded'.

    Returns:
    pd.DataFrame: The DataFrame with an added column containing the one-hot encoded vectors as numpy arrays.
    """
    # One-hot encode the specified column
    encoded_df = pd.get_dummies(df[column_name], prefix=column_name)

    # Convert the one-hot encoded DataFrame to a numpy array
    encoded_np = encoded_df.to_numpy(dtype=int)

    # Assign the numpy arrays back to the DataFrame in a new column
    if new_column_name is None:
        new_column_name = column_name + "_encoded"

    df[new_column_name] = list(encoded_np)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing started")

    id_columns = ["species_name", "upstream200", "chromosome"]
    tpm_columns = [col for col in df.columns if "tpm" in col]

    df = df.dropna(subset=["upstream200"])
    invalid_indices = df[~df["upstream200"].apply(_is_valid_sequence)].index.tolist()
    df = df.drop(invalid_indices)
    df["is_complement"] = df["region"].str.contains("complement")
    df["upstream200"] = df.apply(
        lambda row: (
            complement_dna(row["upstream200"])
            if row["is_complement"]
            else row["upstream200"]
        ),
        axis=1,
    )

    df = df[tpm_columns + id_columns].melt(
        var_name="condition",
        value_name="tpm",
        id_vars=id_columns,
    )
    df.dropna(subset=["tpm"], inplace=True)
    df["condition"] = df["condition"].str.replace("_ge_tpm", "")
    df[["species_abbreviation", "stress_condition_name", "evaluation"]] = df[
        "condition"
    ].str.rsplit("_", n=2, expand=True)

    df.drop(columns=["condition", "species_abbreviation"], inplace=True)

    return df


# Load the data
def get_processed_data(
    project_root_dir: str = None,
    normalize_by_ctrl: bool = True,
    log_transform: bool = True,
    aggregate: str = "mean",
) -> pd.DataFrame:
    """
    Load and preprocess the data for further analysis or model training.

    Parameters:
    - data_df (pd.DataFrame, optional): A DataFrame to load and preprocess. If not provided, the function reads from 'combined_data.csv'.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame ready for analysis or model training.
    """

    if project_root_dir is None:
        project_root_dir = os.getcwd()
    merged_data_path = f"{project_root_dir}/data/merged_data.csv"

    if not os.path.exists(merged_data_path):
        print("The file does not exist.")
        return

    df = pd.read_csv(merged_data_path)

    df["species_name"] = df["species"].map(species_name_to_abb)
    df.drop(columns=["species"], inplace=True)

    if normalize_by_ctrl:
        df = ctrl_normalize(df)

    df = preprocess_data(df)

    id_columns = ["species_name", "upstream200", "chromosome"]

    if aggregate == "mean":
        df = (
            df.groupby(id_columns + ["stress_condition_name"])["tpm"]
            .mean()
            .reset_index()
        )
    elif aggregate == "max":
        df = (
            df.groupby(id_columns + ["stress_condition_name"])["tpm"]
            .max()
            .reset_index()
        )

    if normalize_by_ctrl:
        df = df[df["stress_condition_name"] != "ctrl"]

    if log_transform:
        if normalize_by_ctrl:
            log_tranformation_fn = np.log
        else:
            log_tranformation_fn = np.log1p
        df["tpm"] = df["tpm"].apply(log_tranformation_fn)

    # Encode 'species', 'upstream200' and 'stress_condition' columns as IDs
    max_length = max(df["upstream200"].apply(lambda x: len(x)))
    df["upstream200"] = df["upstream200"].apply(
        lambda seq: onehot_encode_dna(seq, max_length)
    )
    df = one_hot_encode_to_numpy(df, "species_name", new_column_name="species")
    df = one_hot_encode_to_numpy(
        df, "stress_condition_name", new_column_name="stress_condition"
    )
    df.drop(columns=["chromosome"], inplace=True)

    return df


def prepare_datasets(data_df, species_id=-1, size=-1, test_split=0.1):
    """
    Load and preprocess data, and split it into training and testing datasets.

    Parameters:
    - species_id (int, optional): The index of the species ID to filter by. Default is -1 (no filtering).
    - size (int, optional): The number of samples to include. Default is -1 (use all data).
    - val_split (float, optional): The proportion of the data to use for validation. Default is 0.2.
    - test_split (float, optional): The proportion of the data to use for testing. Default is 0.1.
    - data_df (pd.DataFrame, optional): A DataFrame to load and preprocess. If not provided, the function reads from 'combined_data.csv'.

    Returns:
    - train_dataset (SequenceDataset): The training dataset.
    - test_dataset (SequenceDataset): The testing dataset.
    """
    if species_id != -1:
        data_df = data_df[data_df["species_name"].apply(lambda x: x[species_id] == 1)]
    if size != -1:
        size = int(size * 1.39)
        data_df = data_df.sample(size)

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data_df[["species", "stress_condition", "upstream200"]],
        data_df["stress_condition"],
        test_size=test_split,
    )

    # create a dataset
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    return train_dataset, test_dataset


def main():
    processed_data_path = f"{os.getcwd()}/data/processed_data.pkl"
    processed_df = get_processed_data()

    # Save the merged data to a CSV file
    print(f"Data is being saved {processed_data_path}")
    processed_df.to_pickle(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")


if __name__ == "__main__":
    main()

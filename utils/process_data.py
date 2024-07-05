import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utils.helpers import SequenceDataset
from utils.normalizations import ctrl_normalize, get_ctrl_norm, get_mean, get_log_norm

import pandas as pd
from typing import List

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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing started")

    id_columns = ["species", "upstream200", "chromosome"]
    tpm_columns = [col for col in df.columns if "tpm" in col]

    df = df.dropna(subset=["upstream200"])
    invalid_indices = df[~df["upstream_200"].apply(_is_valid_sequence)].index.tolist()
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
        id_vars=["species", "upstream200", "chromosome"],
    )
    df.dropna(subset=["tpm"], inplace=True)
    df["condition"] = df["condition"].str.replace("_ge_tpm", "")
    df[["species_abbreviation", "stress_condition", "evaluation"]] = df[
        "condition"
    ].str.rsplit("_", n=2, expand=True)

    df = df[id_columns + ["stress_condition", "evaluation", "tpm"]]

    max_length = max(df["upstream200"].apply(lambda x: len(x)))
    df["upstream200"] = df["upstream200"].apply(
        lambda seq: onehot_encode_dna(seq, max_length)
    )
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

    if normalize_by_ctrl:
        ctrl_normalize(df, inplace=True)

    df = preprocess_data(df)
    id_columns = ["species", "upstream200", "chromosome"]

    if aggregate == "mean":
        df = df.groupby(id_columns + ["stress_condition"])["tpm"].mean().reset_index()
    elif aggregate == "max":
        df = df.groupby(id_columns + ["stress_condition"])["tpm"].max().reset_index()

    if log_transform:
        df["tpm"] = df["tpm"].apply(np.log1p)

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
        data_df = data_df[data_df["species"].apply(lambda x: x[species_id] == 1)]
    if size != -1:
        size = int(size * 1.39)
        data_df = data_df.sample(size)

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data_df[["species", "stress_name", "upstream200"]],
        data_df["stress"],
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

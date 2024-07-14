import pandas as pd
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utils.helpers import SequenceDataset
from utils.normalizations import get_ctrl_norm, get_mean, get_log_norm

from collections import Counter

def generate_kmers(sequence, k=3):
    if isinstance(sequence, str):
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        return kmers
    else:
        return []
def get_kmer_dataframe(df, k=3):
    kmer_counts = {}

    for i, seq in enumerate(df.upstream200):
        kmers = generate_kmers(seq, k)
        kmer_counts[i] = Counter(kmers)

    kmer_df = pd.DataFrame(kmer_counts).fillna(0).T
    return pd.concat([df, kmer_df], axis=1)


def preprocess_data(df: pd.DataFrame, stress_conditions: set) -> pd.DataFrame:
    print("Preprocessing started")
    # drop rows with missing upstream200 sequences
    df = df.dropna(subset=["upstream200"])
    df.reset_index(drop=True, inplace=True)

    # Add K-mers for k: 3, 6
    df = get_kmer_dataframe(df)
    df = get_kmer_dataframe(df, 6)

    # drop rows with upstream200 sequences that contain anything but A, T, C, G
    df = df[
        df["upstream200"].apply(
            lambda x: set(x).issubset({"A", "T", "C", "G"})
        )
    ]

    mlb = MultiLabelBinarizer()
    # map each species id to a one hot encoding
    df["species_id"] = df["species_id"].apply(lambda x: [x])
    df["species"] = mlb.fit_transform(df["species_id"]).tolist()

    # Drop the columns that are not needed
    df = df.drop(
        columns=[name for name in df.columns if "tpm" in name]
                + ["chromosome", "region", "csv", "species_id"]
    )

    # map each base to one hot encoding
    # One can refactor here to handle different letters
    base_encodings = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }
    longest_sequence = max(df["upstream200"].apply(lambda x: len(x)))
    df["upstream200"] = df["upstream200"].apply(
        lambda x: [base_encodings[base] for base in x]
                  + [[0, 0, 0, 0]] * (longest_sequence - len(x))
    )

    # explode dataset to have one row per stress condition
    df["stress"] = df.apply(
        lambda row: [{stress: row[stress]} for stress in stress_conditions], axis=1
    )
    df = df.drop(
        columns=[name for name in df.columns if name in stress_conditions]
    )

    df = df.explode("stress")
    df["stress_name"] = df["stress"].apply(
        lambda x: list(x.keys())[0]
    )
    df["stress"] = df["stress"].apply(lambda x: list(x.values())[0])

    # one hot encode stress names
    df["stress_name"] = df["stress_name"].apply(lambda x: [x])
    df["stress_name"] = mlb.fit_transform(df["stress_name"]).tolist()

    # drop rows with 0 stress
    df = df[df["stress"] > 0]
    df.reset_index(drop=True, inplace=True)
    return df


# Load the data
def get_processed_data(project_root_dir: str = None,
                       normalize_by_ctrl: bool = True,
                       normalize_by_log: bool = True) -> pd.DataFrame:
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

    if os.path.exists(merged_data_path):
        data_df = pd.read_csv(merged_data_path)
    else:
        print("The file does not exist.")
        return
    averages_df = data_df.copy()
    stress_conditions = set(
        [name.split("_")[0] for name in data_df.columns if "tpm" in name]
    )
    control_condition = "ctrl"
    control_columns = [
        name for name in data_df.columns if control_condition + "_" in name
    ]
    if normalize_by_ctrl:
        stress_conditions.remove(control_condition)

    for stress in stress_conditions:
        stress_columns = [name for name in data_df.columns if stress + "_" in name]
        if normalize_by_ctrl:
            if stress == control_condition:
                continue
            averages_df[f"{stress}"] = get_ctrl_norm(data_df=data_df,
                                                     stress_columns=stress_columns,
                                                     control_columns=control_columns)

        else:
            averages_df[f"{stress}"] = get_mean(data_df=data_df,
                                                stress_columns=stress_columns)

    averages_df = preprocess_data(df=averages_df,
                                  stress_conditions=stress_conditions)

    if normalize_by_log:
        averages_df["stress"] = get_log_norm(df=averages_df, normalize_by_ctrl=normalize_by_ctrl)

    averages_df = averages_df
    return averages_df


def prepare_datasets(data_df,
                     species_id=-1,
                     size=-1,
                     test_split=0.1):
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


def transform_dataframe(df, drop_upstream=True):
    # Convert one-hot encoded lists to numeric values for species
    df['species'] = df['species'].apply(lambda x: x.index(1) if 1 in x else -1)

    # Convert one-hot encoded lists to numeric values for stress_name
    df['stress_name'] = df['stress_name'].apply(lambda x: x.index(1) if 1 in x else -1)

    # Drop the upstream200 column
    if drop_upstream:
        df = df.drop(columns=['upstream200'])

    return df

def main():
    processed_data_path = f"{os.getcwd()}/data/processed_data_kmer_fat.pkl"
    processed_df = get_processed_data()

    # Save the merged data to a CSV file
    print(f"Data is being saved {processed_data_path}")
    processed_df.to_pickle(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")


if __name__ == "__main__":
    main()

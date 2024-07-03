import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from .helpers import SequenceDataset


def calculate_mean(row, stress_cols, control_cols, log_norm):
    values = []
    for stress_col, control_col in zip(stress_cols, control_cols):
        stress_val = row[stress_col]
        control_val = row[control_col]

        # Uncomment below if you want to first take lognorm then norm by crtl
        # if log_norm:
        #     stress_val = np.log(stress_val + 1)
        #     control_val = np.log(control_val + 1)

        # Append if the denominator is not zero and any of the values are not na
        if pd.notna(stress_val) and pd.notna(control_val) and control_val != 0:
            values.append(stress_val / control_val)
    if len(values) > 0:
        return np.mean(values)
    else:
        return 0


# Load the data
def load_dataframe(data_df=None, normalize_by_ctrl=True, log_norm=True):
    """
    Load and preprocess the data for further analysis or model training.

    Parameters:
    - data_df (pd.DataFrame, optional): A DataFrame to load and preprocess. If not provided, the function reads from 'combined_data.csv'.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame ready for analysis or model training.
    """
    if data_df is not None:
        return data_df
    data_df = pd.read_csv("combined_data.csv")

    # get mean of each stress condition
    averages_df = data_df.copy()
    stress_conditions = set(
        [name.split("_")[0] for name in data_df.columns if "TPM" in name]
    )
    control_condition = "Ctrl"
    control_columns = [
        name for name in data_df.columns if control_condition + "_" in name
    ]
    if normalize_by_ctrl:
        stress_conditions.remove(control_condition)

    for stress in stress_conditions:
        if normalize_by_ctrl:
            if stress == control_condition:
                continue
        stress_columns = [name for name in data_df.columns if stress + "_" in name]

        if normalize_by_ctrl:
            averages_df[f"{stress}"] = data_df.apply(
                calculate_mean,
                axis=1,
                stress_cols=stress_columns,
                control_cols=control_columns,
                log_norm=log_norm,
            )
        else:
            averages_df[f"{stress}"] = np.mean(
                [
                    data_df[stress_columns[0]],
                    data_df[stress_columns[1]],
                    data_df[stress_columns[2]],
                ],
                axis=0,
            )

    # Drop the columns that are not needed
    averages_df = averages_df.drop(
        columns=[name for name in averages_df.columns if "TPM" in name]
        + ["Chromosome", "Region", "Species", "Unnamed: 0"]
    )
    # drop rows with missing upstream200 sequences
    averages_df = averages_df.dropna(subset=["upstream200"])
    # drop rows with upstream200 sequences that contain anything but A, T, C, G
    averages_df = averages_df[
        averages_df["upstream200"].apply(
            lambda x: set(x).issubset({"A", "T", "C", "G"})
        )
    ]

    mlb = MultiLabelBinarizer()
    # map each species id to a one hot encoding
    averages_df["Species ID"] = averages_df["Species ID"].apply(lambda x: [x])
    averages_df["Species ID"] = mlb.fit_transform(averages_df["Species ID"]).tolist()

    # map each base to one hot encoding
    base_encodings = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }
    longest_sequence = max(averages_df["upstream200"].apply(lambda x: len(x)))
    averages_df["upstream200"] = averages_df["upstream200"].apply(
        lambda x: [base_encodings[base] for base in x]
        + [[0, 0, 0, 0]] * (longest_sequence - len(x))
    )

    # explode dataset to have one row per stress condition
    averages_df["Stress"] = averages_df.apply(
        lambda row: [{stress: row[stress]} for stress in stress_conditions], axis=1
    )
    averages_df = averages_df.drop(
        columns=[name for name in averages_df.columns if name in stress_conditions]
    )

    averages_df = averages_df.explode("Stress")
    averages_df["Stress_name"] = averages_df["Stress"].apply(
        lambda x: list(x.keys())[0]
    )
    averages_df["Stress"] = averages_df["Stress"].apply(lambda x: list(x.values())[0])

    # one hot encode stress names
    averages_df["Stress_name"] = averages_df["Stress_name"].apply(lambda x: [x])
    averages_df["Stress_name"] = mlb.fit_transform(averages_df["Stress_name"]).tolist()

    # drop rows with 0 stress
    averages_df = averages_df[averages_df["Stress"] > 0]

    if log_norm:
        # log values of stress conditions
        if control_condition:
            # We shouldn't add 1 if we already get the control ratios
            averages_df["Stress"] = averages_df["Stress"].apply(lambda x: np.log(x))
        else:
            averages_df["Stress"] = averages_df["Stress"].apply(lambda x: np.log(x + 1))

    data_df = averages_df
    return averages_df


def load_data(
    species_id=-1,
    size=-1,
    val_split=0.2,
    test_split=0.1,
    data_df=None,
    normalize_by_ctrl=True,
    log_norm=True,
):
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
    data_df = load_dataframe(data_df)

    if species_id != -1:
        data_df = data_df[data_df["Species ID"].apply(lambda x: x[species_id] == 1)]
    if size != -1:
        size = int(size * 1.39)
        data_df = data_df.sample(size)

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data_df[["Species ID", "Stress_name", "upstream200"]],
        data_df["Stress"],
        test_size=test_split,
    )

    # create a dataset
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    return train_dataset, test_dataset

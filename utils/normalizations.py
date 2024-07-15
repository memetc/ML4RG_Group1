import pandas as pd
import numpy as np


def calculate_mean(row, stress_cols, control_cols):
    """
    Calculate the mean of the ratios of stress values to control values.

    This function computes the mean of the ratios between stress columns and control columns for a given row.
    It only includes ratios where both stress and control values are non-NA and the control value is not zero.

    Parameters:
    row (pd.Series): A row from a DataFrame.
    stress_cols (list of str): List of column names representing stress conditions.
    control_cols (list of str): List of column names representing control conditions.

    Returns:
    float: The mean of the ratios of stress values to control values for the given row.
           Returns 0 if no valid ratios are found.
    """
    normalized_values, original_values = [], []
    for stress_col, control_col in zip(stress_cols, control_cols):
        stress_val = row[stress_col]
        control_val = row[control_col]

        # Append if the denominator is not zero and any of the values are not na
        if pd.notna(stress_val) and pd.notna(control_val):
            normalized_values.append(np.log1p(stress_val) - np.log1p(control_val))
            original_values.append(stress_val)
    if len(normalized_values) > 0:
        # TODO
        return np.mean(normalized_values), np.mean(normalized_values)
    else:
        return np.nan, np.nan


def get_ctrl_norm(data_df, stress_columns, control_columns):
    return data_df.apply(
        calculate_mean,
        axis=1,
        stress_cols=stress_columns,
        control_cols=control_columns,
        result_type="expand",
    )


def get_mean(data_df, stress_columns):
    """
    Normalize data by control columns.

    Parameters:
    data_df (pd.DataFrame): DataFrame containing the data to be normalized.
    stress_columns (list of str): List of column names representing stress conditions.
    control_columns (list of str): List of column names representing control conditions.

    Returns:
    pd.Series: A series containing the normalized values for each row in the data frame.
    """

    np.mean(
        [
            data_df[stress_columns[0]],
            data_df[stress_columns[1]],
            data_df[stress_columns[2]],
        ],
        axis=0,
    )


# # Function to calculate log normalization conditionally
# def get_log_norm(df: pd.DataFrame, normalize_by_ctrl: bool):
#     log_stress = df.apply(
#         lambda row: (
#             np.log(row["stress"])
#             if row["is_normalized"] and normalize_by_ctrl
#             else (np.log(row["stress"] + 1) if row["is_normalized"] else row["stress"])
#         ),
#         axis=1,
#     )
#     df['stress'] = log_stress
#     return df

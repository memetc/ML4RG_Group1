import pandas as pd
import numpy as np

def calculate_mean(row, stress_cols, control_cols):
    values = []
    for stress_col, control_col in zip(stress_cols, control_cols):
        stress_val = row[stress_col]
        control_val = row[control_col]

        # Append if the denominator is not zero and any of the values are not na
        if pd.notna(stress_val) and pd.notna(control_val) and control_val != 0:
            values.append(stress_val / control_val)
    if len(values) > 0:
        return np.mean(values)
    else:
        return 0


def get_ctrl_norm(data_df, stress_columns, control_columns):
    return data_df.apply(
        calculate_mean,
        axis=1,
        stress_cols=stress_columns,
        control_cols=control_columns,
    )

def get_mean(data_df, stress_columns):
    np.mean(
        [
            data_df[stress_columns[0]],
            data_df[stress_columns[1]],
            data_df[stress_columns[2]],
        ],
        axis=0,
    )


def get_log_norm(df: pd.DataFrame, normalize_by_ctrl: bool):
    # log values of stress conditions
    return df["stress"].apply(lambda x: np.log(x)) if normalize_by_ctrl else df["stress"].apply(lambda x: np.log(x + 1))

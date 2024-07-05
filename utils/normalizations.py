import pandas as pd
import numpy as np

from collections import defaultdict

CONTROL_CONDITION_KEY = "ctrl"


def ctrl_normalize(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Normalizes each <SPECIES_NAME>_<CONDITION>_<REPETITION> column by the corresponding <SPECIES_NAME>_ctrl_<REPETITION> column.

    This function processes a DataFrame by dividing the TPM (transcripts per million) values of each condition column
    by the corresponding control column for each species and repetition. The normalization is done only if both the
    condition and control columns exist in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing TPM columns with names following the pattern
                       <SPECIES_NAME>_<CONDITION>_<REPETITION>_ge_tpm.

    Returns:
    pd.DataFrame: The DataFrame with normalized TPM values.
    """
    # Identify the unique species and conditions
    if inplace:
        output_df = df
    else:
        output_df = df.copy()

    tpm_columns = defaultdict(lambda: defaultdict(str))
    for col in output_df.columns:
        if "tpm" in col:
            species, stress, rep = col.split("_")[:3]
            if stress != CONTROL_CONDITION_KEY:
                tpm_columns[species][stress] = rep

    # Divide each <SPECIES_NAME>_<CONDITION>_<REPETITION> by the corresponding <SPECIES_NAME>_ctrl_<REPETITION>
    for species, cond2rep in tpm_columns.items():
        for stress, rep in cond2rep.items():
            condition_col = f"{species}_{stress}_{rep}_ge_tpm"
            control_col = f"{species}_{CONTROL_CONDITION_KEY}_{rep}_ge_tpm"
            if control_col in output_df.columns and condition_col in output_df.columns:
                output_df[condition_col] = np.where(
                    output_df[control_col].isna(),
                    output_df[condition_col],
                    np.where(
                        output_df[control_col] == 0,
                        output_df[condition_col],
                        output_df[condition_col] / output_df[control_col],
                    ),
                )
            if control_col in output_df.columns:
                output_df.drop(columns=[control_col], inplace=True)

    if not inplace:
        return output_df

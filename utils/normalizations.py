import pandas as pd
import numpy as np
from collections import defaultdict

CONTROL_CONDITION_KEY = "ctrl"


def ctrl_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes each <CONDITION>_<REPETITION> column by the corresponding <CONDITION>_<REPETITION> column for each species.

    This function processes a DataFrame by dividing the TPM (transcripts per million) values of each condition column
    by the corresponding control column for each species and repetition. The normalization is done only if both the
    condition and control columns exist in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing species names and TPM columns with names following the pattern
                       <CONDITION>_<REPETITION>_ge_tpm.

    Returns:
    pd.DataFrame: The DataFrame with normalized TPM values.
    """
    species_list = df["species_name"].unique()

    # Identify the unique conditions and repetitions for each species
    tpm_columns = defaultdict(lambda: defaultdict(str))
    for col in df.columns:
        if "tpm" in col:
            stress, rep = col.split("_")[:2]
            if stress != CONTROL_CONDITION_KEY:
                for species in species_list:
                    tpm_columns[species][stress] = rep

    # Divide each <CONDITION>_<REPETITION> by the corresponding <CONDITION>_<REPETITION> for each species
    for species in species_list:
        for stress, rep in tpm_columns[species].items():
            condition_col = f"{stress}_{rep}_ge_tpm"
            control_col = f"{CONTROL_CONDITION_KEY}_{rep}_ge_tpm"
            if control_col in df.columns and condition_col in df.columns:
                mask = df["species_name"] == species
                df.loc[mask, condition_col] = np.where(
                    df.loc[mask, control_col].isna(),
                    df.loc[mask, condition_col],
                    np.where(
                        df.loc[mask, control_col] == 0,
                        df.loc[mask, condition_col],
                        df.loc[mask, condition_col] / df.loc[mask, control_col],
                    ),
                )
            if control_col in df.columns:
                df.drop(columns=[control_col], inplace=True)

    return df

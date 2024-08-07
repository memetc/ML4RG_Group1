import pandas as pd
import os
import re
from typing import List, Dict
from utils.helpers import species_abb_to_name, stress_columns, species_name_to_abb



def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format column name by replacing spaces and special characters with underscores and converting to lowercase.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be renamed.

    Returns:
    df (pd.DataFrame): Renamed DataFrame.
    """
    # Create a mapping dictionary from stress columns
    prefixes = list(species_abb_to_name.keys())
    pattern = re.compile(r'(' + '|'.join(prefixes) + r')_(\w+ \(\w+\) - TPM)')
    mapping = {f"{prefix}_{col.split()[0]}": col for col in stress_columns for prefix in prefixes}

    new_columns = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            key = f"{match.group(1)}_{match.group(2).split()[0]}"
            if key in mapping:
                new_columns.append(mapping[key])
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)

    df.columns = new_columns

    if 'Aggregatibacter actinomycetemcomitans D7S-1' in set(df['Species']):
        df['species_id'] = "AGGA"
    else:
        df['species_id'] = df['Species'].map(species_name_to_abb)
    return df

def clean_column_name(col_name: str) -> str:
    """
    Clean and format column name by replacing spaces and special characters with underscores and converting to lowercase.

    Parameters:
    col_name (str): The original column name.

    Returns:
    str: The cleaned and formatted column name.
    """
    col_name = col_name.replace("_", " ")
    col_name = col_name.replace("-", " ")
    col_name = re.sub(r"[^\w\s]", " ", col_name)
    col_name = re.sub(r"\s+", " ", col_name)
    col_name = re.sub(r"[^\w\s]", "", col_name)
    col_name = col_name.replace(" ", "_")
    cleaned_name = col_name.lower()
    return cleaned_name


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename all columns of the DataFrame to a more convenient format.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be renamed.

    Returns:
    pd.DataFrame: DataFrame with renamed columns.
    """
    new_columns = {col: clean_column_name(col) for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    return df


def read_csv_file_with_filename(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame, adding a column with the CSV filename.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The DataFrame created from the CSV file with an additional column 'csv'
               containing the filename. If parsing fails, returns an empty DataFrame.
    """
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8", low_memory=False)
        df["csv"] = os.path.basename(file_path)
    except pd.errors.ParserError:
        print(f"Error parsing {file_path}")
        return pd.DataFrame()
    return df


def get_expression_data(data_path: str) -> pd.DataFrame:
    """
    Get the expression data by reading all CSV files in the specified directory,
    cleaning column names, and selecting specific columns.

    Args:
    data_path (str): The path to the directory containing the CSV files.

    Returns:
    DataFrame: A concatenated DataFrame containing data from all CSV files in the directory.
    """
    all_files: List[str] = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if file.endswith(".csv")
    ]
    df_list: List[pd.DataFrame] = [
        read_csv_file_with_filename(file) for file in all_files
    ]
    df_list = [standardize_columns(df) for df in df_list]
    expression_df: pd.DataFrame = pd.concat(df_list, ignore_index=True)
    expression_df.reset_index(drop=True, inplace=True)
    expression_df = rename_columns(expression_df)
    tpm_columns = [col for col in expression_df.columns if "tpm" in col]
    expression_df = expression_df[
        ["species", "csv", "chromosome", "region", "species_id"] + tpm_columns
    ]
    return expression_df


def get_upstream_data(file_path: str) -> pd.DataFrame:
    """
    Get the upstream data from an Excel file, rename columns, replace values, and ensure consistency with the expression data.

    Args:
    file_path (str): The path to the Excel file containing the upstream data.
    expression_df (DataFrame): The DataFrame containing the expression data.

    Returns:
    DataFrame: The processed upstream DataFrame.
    """
    if file_path.endswith(".xlsx"):
        upstream_df: pd.DataFrame = pd.read_excel(file_path)
    elif file_path.endswith(".tsv"):
        upstream_df: pd.DataFrame =  pd.read_csv(file_path, sep='\t')
    upstream_df.rename(columns={"contig": "chromosome"}, inplace=True)
    replacement_dict: Dict[str, str] = {
        "Staphylococcus��aureus MRSA252.csv": "Staphylococcus\xa0aureus MRSA252.csv",
        "Staphylococcus��aureus MSSA476.csv": "Staphylococcus\xa0aureus MSSA476.csv",
        "Staphylococcus��epidermidis 1457.csv": "Staphylococcus\xa0epidermidis 1457.csv",
    }
    upstream_df["csv"] = upstream_df["csv"].replace(replacement_dict)

    return upstream_df


def get_merged_data(
    expression_df: pd.DataFrame, upstream_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the expression data and upstream data on the columns 'csv', 'region', and 'chromosome'.

    Args:
    expression_df (DataFrame): The DataFrame containing the expression data.
    upstream_df (DataFrame): The DataFrame containing the upstream data.

    Returns:
    DataFrame: The merged DataFrame.
    """
    merged_df: pd.DataFrame = pd.merge(
        expression_df, upstream_df, on=["csv", "region", "chromosome"], how="left"
    )
    return merged_df


def main():
    # Paths
    expression_data_path = f"{os.getcwd()}/data/data_expression"
    upstream_data_path = (
        f"{os.getcwd()}/data/data_sequences_upstream/upstream_sequences.tsv"
    )
    merged_data_path = f"{os.getcwd()}/data/merged_data.csv"

    # Get expression data
    expression_df = get_expression_data(expression_data_path)

    # Get upstream data
    upstream_df = get_upstream_data(upstream_data_path)

    # Get merged data
    merged_df = get_merged_data(expression_df, upstream_df)

    # Save the merged data to a CSV file
    merged_df.to_csv(merged_data_path, index=False)
    print(f"Merged data saved to {merged_data_path}")


if __name__ == "__main__":
    main()

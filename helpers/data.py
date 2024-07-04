import os
import pandas as pd


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter=";", encoding="utf-8", low_memory=False)
    except pd.errors.ParserError:
        print(f"Error parsing {file_path}")
        return pd.DataFrame()


def read_csv_file_with_filename(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8", low_memory=False)
        df["csv"] = os.path.basename(file_path)
    except pd.errors.ParserError:
        print(f"Error parsing {file_path}")
        return pd.DataFrame()
    return df


def clean_column_name(col_name):
    """
    clean and format column name by replacing spaces and special characters with underscores and converting to lowercase.

    Parameters:
    col_name (str): The original column name.

    Returns:
    str: The cleaned and formatted column name.
    """
    col_name = col_name.replace("_", " ")
    col_name = col_name.replace("-", " ")
    col_name = re.sub(r"[^\w\s]", " ", col_name)
    col_name = re.sub(r"\s+", " ", col_name)
    # Replace non-word characters (except for spaces) with nothing
    col_name = re.sub(r"[^\w\s]", "", col_name)
    # Replace spaces with underscores
    col_name = col_name.replace(" ", "_")
    # Convert to lowercase
    cleaned_name = col_name.lower()
    return cleaned_name


def rename_columns(df):
    """
    Rename all columns of the DataFrame to a more convenient format.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be renamed.

    Returns:
    pd.DataFrame: DataFrame with renamed columns.
    """
    # Create a dictionary to map old column names to new column names
    new_columns = {col: clean_column_name(col) for col in df.columns}

    # Rename columns in the DataFrame
    df.rename(columns=new_columns, inplace=True)

    return df


def complement_dna(sequence):
    return "".join(COMPLEMENT[base] for base in sequence)


def get_expression_df(data_path=None):
    if data_path is None:
        data_path = f"{os.getcwd()}/data/data_expression"

    all_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if file.endswith(".csv")
    ]

    df_list = [read_csv_file_with_filename(file) for file in all_files]
    expression_df = pd.concat(df_list, ignore_index=True)
    expression_df.reset_index(drop=True, inplace=True)

    # Rename all columns to a more convenient format
    expression_df = rename_columns(expression_df)

    tpm_columns = [col for col in expression_df.columns if "tpm" in col]
    expression_df = expression_df[
        ["species", "csv", "chromosome", "region"] + tpm_columns
    ]
    return expression_df


def get_upstream_df(data_path=None):
    if data_path is None:
        data_path = (
            f"{os.getcwd()}/data/data_sequences_upstream/upstream_sequences.xlsx"
        )

    upstream_df = pd.read_excel(data_path)
    upstream_df.rename(columns={"contig": "chromosome"}, inplace=True)

    replacement_dict = {
        "Staphylococcus��aureus MRSA252.csv": "Staphylococcus\xa0aureus MRSA252.csv",
        "Staphylococcus��aureus MSSA476.csv": "Staphylococcus\xa0aureus MSSA476.csv",
        "Staphylococcus��epidermidis 1457.csv": "Staphylococcus\xa0epidermidis 1457.csv",
    }
    upstream_df["csv"] = upstream_df["csv"].replace(replacement_dict)

    return upstream_df


def get_and_save_data(
    expression_data_path=None, upstream_data_path=None, save_path=None
):
    expression_df = get_expression_df(expression_data_path)
    upstream_df = get_upstream_df(upstream_data_path)

    assert len(set(upstream_df["csv"]).difference(set(expression_df["csv"]))) == 0
    assert (
        len(
            set(upstream_df[["csv", "region"]]).difference(
                set(expression_df[["csv", "region"]])
            )
        )
        == 0
    )

    merged_df = pd.merge(expression_df, upstream_df, on=["csv", "region"], how="left")
    merged_df["upstream200"].fillna("", inplace=True)
    # Define the valid characters
    valid_chars = set("ATCG")

    # Function to check if a sequence is valid
    def is_valid_sequence(seq):
        return len(seq) > 0 and set(seq).issubset(valid_chars)

    # Apply the function to each sequence and get indices of invalid sequences
    invalid_indices = merged_df[
        ~merged_df["upstream200"].apply(is_valid_sequence)
    ].index.tolist()

    # Drop the invalid sequences
    merged_df = merged_df.drop(invalid_indices)
    merged_df.dropna(subset=["species"], inplace=True)
    merged_df["is_complement"] = merged_df["region"].str.contains("complement")

    # Apply the complement_dna function to the upstream200 column where is_complement is True
    merged_df["upstream200"] = merged_df.apply(
        lambda row: (
            complement_dna(row["upstream200"])
            if row["is_complement"]
            else row["upstream200"]
        ),
        axis=1,
    )
    tpm_columns = [col for col in merged_df.columns if "tpm" in col]
    merged_df = merged_df[["species", "upstream200"] + tpm_columns]
    merged_df.reset_index(drop=True, inplace=True)
    if save_path is None:
        save_path = f"{os.getcwd()}/data/merged.csv"
    merged_df.to_csv(save_path)
    return merged_df


def get_and_save_mean_df(data_path=None, save_path=None):
    if data_path is None:
        merged_df = get_and_save_data()
    else:
        merged_df = pd.read_csv(data_path, index_col=0)

    melted_df = merged_df.melt(
        var_name="condition", value_name="tpm", id_vars=["species", "upstream200"]
    )
    melted_df.dropna(subset=["tpm"], inplace=True)
    melted_df["condition"] = melted_df["condition"].str.replace("_ge_tpm", "")
    melted_df[["stress_condition", "evaluation"]] = melted_df["condition"].str.rsplit(
        "_", n=1, expand=True
    )
    melted_df.drop(columns=["condition"], inplace=True)

    # Calculating the mean for each 'stress_condition'
    mean_df = (
        melted_df.groupby(["species", "upstream200", "stress_condition"])["tpm"]
        .mean()
        .reset_index()
    )
    mean_df.rename(columns={"tpm": "mean_tpm"}, inplace=True)

    if save_path is None:
        save_path = f"{os.getcwd()}/data/mean_tpm.csv"

    mean_df.to_csv(save_path)
    return mean_df

"""
loader.py

Responsibilities:
  - load_data()  : reads the raw survey CSV from disk into a DataFrame
  - clean_data() : filters, normalises, and prepares data for analysis

The Stack Overflow Developer Survey stores multiple CI/CD tool selections
as semicolon-separated strings in a single cell, e.g.:
    "GitHub Actions;Jenkins;CircleCI"
clean_data() splits these into Python lists so downstream analysis can
count individual tool occurrences correctly.
"""

import pandas as pd


# Columns required for this analysis — anything else is discarded early
# to keep memory usage low on the full 65k-row dataset
REQUIRED_COLUMNS = [
    "ResponseId",
    "ToolsTechHaveWorkedWith",  # This is the actual column name in 2024 survey
    "OrgSize",
    "DevType",
    "Country",
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Stack Overflow Developer Survey CSV from disk.

    Parameters
    ----------
    filepath : str
        Path to survey_results_public.csv

    Returns
    -------
    pd.DataFrame
        Raw survey data with only the required columns retained.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    try:
        df = pd.read_csv(filepath, usecols=lambda col: col in REQUIRED_COLUMNS)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Survey data file not found at: {filepath}\n"
            "Download it from https://survey.stackoverflow.co/ and place it in data/"
        )

    # Retain only the columns we need, in a consistent order
    present = [col for col in REQUIRED_COLUMNS if col in df.columns]
    df = df[present]

    # Rename the CI/CD tools column to the expected name
    if "ToolsTechHaveWorkedWith" in df.columns:
        df = df.rename(columns={"ToolsTechHaveWorkedWith": "CICDTools"})

    return df.reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise the raw survey DataFrame.

    Steps applied:
      1. Work on a copy — original DataFrame is never mutated.
      2. Drop rows where CICDTools is null — these cannot contribute to
         tool frequency analysis.
      3. Split the semicolon-separated CICDTools string into a list of
         individual tool names, stripping whitespace from each entry.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for analysis.

    Raises
    ------
    KeyError
        If the CICDTools column is not present in the input DataFrame.
    """
    if "CICDTools" not in df.columns:
        raise KeyError(
            "Column 'CICDTools' not found. "
            "Ensure the DataFrame was loaded with load_data()."
        )

    cleaned = df.copy()

    # Drop rows with no CI/CD tool response
    cleaned = cleaned.dropna(subset=["CICDTools"]).reset_index(drop=True)

    # Split semicolon-delimited strings into lists, stripping whitespace
    cleaned["CICDTools"] = cleaned["CICDTools"].apply(
        lambda raw: [tool.strip() for tool in raw.split(";")]
    )

    return cleaned

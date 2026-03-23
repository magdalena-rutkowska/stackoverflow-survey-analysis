"""
analysis.py

Responsibilities:
  - get_tool_counts()          : overall frequency of each CI/CD tool
  - get_adoption_by_org_size() : tool counts broken down by company size
  - get_adoption_by_devtype()  : tool counts broken down by developer role
  - get_top_tools()            : returns the N most-used tools

All functions expect a cleaned DataFrame as produced by loader.clean_data(),
where CICDTools is already a list of strings per row.
"""

import pandas as pd


def _explode_tools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper: explodes the CICDTools list column so each tool
    occupies its own row, enabling straightforward groupby and count
    operations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame where CICDTools is a list per row.

    Returns
    -------
    pd.DataFrame
        Exploded DataFrame with one tool per row.

    Raises
    ------
    KeyError
        If CICDTools column is absent.
    """
    if "CICDTools" not in df.columns:
        raise KeyError("Column 'CICDTools' not found in DataFrame.")

    return df.explode("CICDTools").rename(columns={"CICDTools": "Tool"})


def get_tool_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count how many survey respondents reported using each CI/CD tool.

    Because one respondent can use multiple tools, a single row may
    contribute to several tool counts — this is intentional.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from loader.clean_data().

    Returns
    -------
    pd.Series
        Tool names as index, integer counts as values, sorted descending.

    Raises
    ------
    KeyError
        If CICDTools column is absent.
    """
    if "CICDTools" not in df.columns:
        raise KeyError("Column 'CICDTools' not found in DataFrame.")

    exploded = _explode_tools(df)
    counts = exploded["Tool"].value_counts()
    counts = counts.astype("int64")

    return counts


def get_adoption_by_org_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count CI/CD tool usage broken down by organisation size.

    Useful for identifying whether enterprise organisations (10,000+
    employees) favour different tooling than smaller companies.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from loader.clean_data().

    Returns
    -------
    pd.DataFrame
        Index: OrgSize categories.
        Columns: CI/CD tool names.
        Values: integer counts of respondents using each tool per size band.

    Raises
    ------
    KeyError
        If CICDTools or OrgSize columns are absent.
    """
    if "CICDTools" not in df.columns:
        raise KeyError("Column 'CICDTools' not found in DataFrame.")
    if "OrgSize" not in df.columns:
        raise KeyError("Column 'OrgSize' not found in DataFrame.")

    exploded = _explode_tools(df)

    pivot = (
        exploded.groupby(["OrgSize", "Tool"])
        .size()
        .unstack(fill_value=0)
    )

    return pivot


def get_adoption_by_devtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count CI/CD tool usage broken down by developer role (DevType).

    Useful for identifying whether SREs and DevOps specialists show
    different tool preferences compared to general developers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from loader.clean_data().

    Returns
    -------
    pd.DataFrame
        Index: DevType categories.
        Columns: CI/CD tool names.
        Values: integer counts of respondents using each tool per role.

    Raises
    ------
    KeyError
        If CICDTools or DevType columns are absent.
    """
    if "CICDTools" not in df.columns:
        raise KeyError("Column 'CICDTools' not found in DataFrame.")
    if "DevType" not in df.columns:
        raise KeyError("Column 'DevType' not found in DataFrame.")

    exploded = _explode_tools(df)

    pivot = (
        exploded.groupby(["DevType", "Tool"])
        .size()
        .unstack(fill_value=0)
    )

    return pivot


def get_top_tools(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    Return the N most-used CI/CD tools across all survey respondents.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from loader.clean_data().
    n : int, optional
        Number of top tools to return. Defaults to 10.
        If n exceeds the total number of distinct tools, all tools
        are returned without raising an error.

    Returns
    -------
    pd.Series
        Top N tools sorted by count descending.

    Raises
    ------
    KeyError
        If CICDTools column is absent.
    ValueError
        If n is zero or negative.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")

    counts = get_tool_counts(df)

    return counts.head(n)

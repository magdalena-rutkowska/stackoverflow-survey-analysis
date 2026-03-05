"""
Tests for src/analysis.py

Covers:
  - get_tool_counts()         : frequency of each CI/CD tool across all responses
  - get_adoption_by_org_size(): CI/CD tool counts broken down by company size
  - get_adoption_by_devtype() : CI/CD tool counts broken down by developer role
  - get_top_tools()           : returns the N most-used tools

All tests use an in-memory fixture that mirrors the cleaned output of
loader.clean_data() — i.e. CICDTools is already a list, not a raw string.
"""

import pandas as pd
import pytest

from src.analysis import (
    get_tool_counts,
    get_adoption_by_org_size,
    get_adoption_by_devtype,
    get_top_tools,
)


# ---------------------------------------------------------------------------
# Shared fixture — mirrors cleaned output from loader.clean_data()
# ---------------------------------------------------------------------------

@pytest.fixture
def cleaned_df():
    """
    Small cleaned DataFrame where CICDTools is already split into lists,
    as clean_data() in loader.py would produce.
    """
    return pd.DataFrame({
        "ResponseId": [1, 2, 3, 4, 5, 6],
        "CICDTools": [
            ["GitHub Actions", "Jenkins"],
            ["GitLab CI/CD"],
            ["GitHub Actions", "CircleCI"],
            ["Jenkins"],
            ["GitHub Actions"],
            ["Jenkins", "CircleCI"],
        ],
        "OrgSize": [
            "100 to 499 employees",
            "1,000 to 4,999 employees",
            "100 to 499 employees",
            "10,000 or more employees",
            "1,000 to 4,999 employees",
            "10,000 or more employees",
        ],
        "DevType": [
            "DevOps specialist",
            "Site Reliability Engineer",
            "Developer, back-end",
            "DevOps specialist",
            "Site Reliability Engineer",
            "Developer, back-end",
        ],
        "Country": ["UK", "US", "Germany", "India", "Canada", "UK"],
    })


# ---------------------------------------------------------------------------
# get_tool_counts()
# ---------------------------------------------------------------------------

class TestGetToolCounts:

    def test_returns_a_series(self, cleaned_df):
        """get_tool_counts() must return a pandas Series."""
        result = get_tool_counts(cleaned_df)

        assert isinstance(result, pd.Series)

    def test_series_is_not_empty(self, cleaned_df):
        """Result must contain at least one tool entry."""
        result = get_tool_counts(cleaned_df)

        assert len(result) > 0

    def test_all_tools_are_counted(self, cleaned_df):
        """
        Every distinct tool name that appears in CICDTools across all
        rows must appear in the result index.
        """
        result = get_tool_counts(cleaned_df)

        expected_tools = {"GitHub Actions", "Jenkins", "GitLab CI/CD", "CircleCI"}
        assert expected_tools.issubset(set(result.index))

    def test_counts_are_correct(self, cleaned_df):
        """
        GitHub Actions appears in rows 1, 3, 5 = count of 3.
        Jenkins appears in rows 1, 4, 6 = count of 3.
        CircleCI appears in rows 3, 6 = count of 2.
        GitLab CI/CD appears in row 2 = count of 1.
        """
        result = get_tool_counts(cleaned_df)

        assert result["GitHub Actions"] == 3
        assert result["Jenkins"] == 3
        assert result["CircleCI"] == 2
        assert result["GitLab CI/CD"] == 1

    def test_result_is_sorted_descending(self, cleaned_df):
        """
        Counts must be sorted highest to lowest to make visualisation
        straightforward without additional transformation.
        """
        result = get_tool_counts(cleaned_df)

        assert result.is_monotonic_decreasing

    def test_counts_are_integer_type(self, cleaned_df):
        """Count values must be integers, not floats."""
        result = get_tool_counts(cleaned_df)

        assert result.dtype in ["int64", "int32"]

    def test_raises_on_missing_cicdtools_column(self, cleaned_df):
        """
        If the DataFrame does not contain a CICDTools column,
        a KeyError or ValueError must be raised — not a silent wrong result.
        """
        df_missing_col = cleaned_df.drop(columns=["CICDTools"])

        with pytest.raises((KeyError, ValueError)):
            get_tool_counts(df_missing_col)


# ---------------------------------------------------------------------------
# get_adoption_by_org_size()
# ---------------------------------------------------------------------------

class TestGetAdoptionByOrgSize:

    def test_returns_a_dataframe(self, cleaned_df):
        """get_adoption_by_org_size() must return a pandas DataFrame."""
        result = get_adoption_by_org_size(cleaned_df)

        assert isinstance(result, pd.DataFrame)

    def test_org_size_values_are_in_index_or_columns(self, cleaned_df):
        """
        Each distinct OrgSize value must appear somewhere in the result
        so chart code can label axes correctly.
        """
        result = get_adoption_by_org_size(cleaned_df)

        org_sizes = cleaned_df["OrgSize"].dropna().unique()
        for size in org_sizes:
            assert size in result.index or size in result.columns

    def test_tool_names_appear_in_result(self, cleaned_df):
        """
        CI/CD tool names must appear in the result so adoption rates
        per organisation size can be compared.
        """
        result = get_adoption_by_org_size(cleaned_df)

        # At least one known tool must be represented
        known_tools = {"GitHub Actions", "Jenkins", "GitLab CI/CD", "CircleCI"}
        result_labels = set(result.index) | set(result.columns)
        assert len(known_tools & result_labels) > 0

    def test_result_contains_no_negative_values(self, cleaned_df):
        """Count values must never be negative."""
        result = get_adoption_by_org_size(cleaned_df)

        assert (result.values >= 0).all()

    def test_raises_on_missing_orgsize_column(self, cleaned_df):
        """
        A missing OrgSize column must raise KeyError or ValueError
        rather than returning an empty or malformed result silently.
        """
        df_missing_col = cleaned_df.drop(columns=["OrgSize"])

        with pytest.raises((KeyError, ValueError)):
            get_adoption_by_org_size(df_missing_col)


# ---------------------------------------------------------------------------
# get_adoption_by_devtype()
# ---------------------------------------------------------------------------

class TestGetAdoptionByDevtype:

    def test_returns_a_dataframe(self, cleaned_df):
        """get_adoption_by_devtype() must return a pandas DataFrame."""
        result = get_adoption_by_devtype(cleaned_df)

        assert isinstance(result, pd.DataFrame)

    def test_devtype_values_appear_in_result(self, cleaned_df):
        """
        Each distinct DevType from the input must appear in the result
        so role-based comparisons can be made.
        """
        result = get_adoption_by_devtype(cleaned_df)

        dev_types = cleaned_df["DevType"].dropna().unique()
        for dtype in dev_types:
            assert dtype in result.index or dtype in result.columns

    def test_result_contains_no_negative_values(self, cleaned_df):
        """Count values must never be negative."""
        result = get_adoption_by_devtype(cleaned_df)

        assert (result.values >= 0).all()

    def test_raises_on_missing_devtype_column(self, cleaned_df):
        """
        A missing DevType column must raise KeyError or ValueError
        rather than silently returning a wrong result.
        """
        df_missing_col = cleaned_df.drop(columns=["DevType"])

        with pytest.raises((KeyError, ValueError)):
            get_adoption_by_devtype(df_missing_col)


# ---------------------------------------------------------------------------
# get_top_tools()
# ---------------------------------------------------------------------------

class TestGetTopTools:

    def test_returns_a_series(self, cleaned_df):
        """get_top_tools() must return a pandas Series."""
        result = get_top_tools(cleaned_df, n=3)

        assert isinstance(result, pd.Series)

    def test_returns_correct_number_of_tools(self, cleaned_df):
        """Result must contain exactly n tools when n < total distinct tools."""
        result = get_top_tools(cleaned_df, n=3)

        assert len(result) == 3

    def test_returns_all_tools_when_n_exceeds_total(self, cleaned_df):
        """
        If n is larger than the number of distinct tools, return all tools
        rather than raising an error.
        """
        result = get_top_tools(cleaned_df, n=100)

        distinct_tools = {t for tools in cleaned_df["CICDTools"] for t in tools}
        assert len(result) == len(distinct_tools)

    def test_top_tools_are_highest_count(self, cleaned_df):
        """
        The tools returned must be the ones with the highest counts,
        not an arbitrary selection.
        """
        all_counts = get_tool_counts(cleaned_df)
        top_n = get_top_tools(cleaned_df, n=2)

        # The top 2 by count must match the first 2 of the full sorted series
        assert list(top_n.index) == list(all_counts.head(2).index)

    def test_default_n_is_ten(self, cleaned_df):
        """
        When called without an explicit n, get_top_tools() should default
        to returning 10 tools (or all tools if fewer than 10 exist).
        """
        result = get_top_tools(cleaned_df)

        distinct_tools = {t for tools in cleaned_df["CICDTools"] for t in tools}
        expected_count = min(10, len(distinct_tools))
        assert len(result) == expected_count

    def test_raises_on_invalid_n(self, cleaned_df):
        """n must be a positive integer — zero or negative must raise ValueError."""
        with pytest.raises(ValueError):
            get_top_tools(cleaned_df, n=0)

        with pytest.raises(ValueError):
            get_top_tools(cleaned_df, n=-1)

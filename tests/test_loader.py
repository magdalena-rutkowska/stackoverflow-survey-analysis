"""
Tests for src/loader.py

Covers:
  - load_data()   : reads the CSV and returns a DataFrame
  - clean_data()  : filters, normalises, and prepares the DataFrame
                    for downstream analysis

All tests use a small in-memory fixture that mirrors the real survey
schema, so the test suite runs without the full dataset present.
"""

import pandas as pd
import pytest

from src.loader import load_data, clean_data


# ---------------------------------------------------------------------------
# Shared fixture — a minimal DataFrame that mirrors the real survey columns
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_survey_df():
    """
    Mimics a small slice of survey_results_public.csv.
    CICDTools uses semicolons to separate multiple values, exactly as the
    real dataset does.
    """
    return pd.DataFrame({
        "ResponseId": [1, 2, 3, 4, 5],
        "ToolsTechHaveWorkedWith": [
            "GitHub Actions;Jenkins",
            "GitLab CI/CD",
            None,                           # missing — should be dropped
            "GitHub Actions;CircleCI",
            "Jenkins",
        ],
        "OrgSize": [
            "100 to 499 employees",
            "1,000 to 4,999 employees",
            "20 to 99 employees",
            None,                           # missing — should be handled
            "10,000 or more employees",
        ],
        "DevType": [
            "Developer, back-end;DevOps specialist",
            "Site Reliability Engineer",
            "Developer, full-stack",
            "Engineer, data",
            None,                           # missing — should be handled
        ],
        "YearsCodePro": ["5", "10", "notaNumber", "3", "15"],
        "Country": ["UK", "US", "Germany", "India", "Canada"],
    })


@pytest.fixture
def loaded_df():
    """
    Mimics the output of load_data() — ToolsTechHaveWorkedWith has already
    been renamed to CICDTools, but values are still raw semicolon strings.
    """
    return pd.DataFrame({
        "ResponseId": [1, 2, 3, 4, 5],
        "CICDTools": [
            "GitHub Actions;Jenkins",
            "GitLab CI/CD",
            None,                           # missing — should be dropped
            "GitHub Actions;CircleCI",
            "Jenkins",
        ],
        "OrgSize": [
            "100 to 499 employees",
            "1,000 to 4,999 employees",
            "20 to 99 employees",
            None,                           # missing — should be handled
            "10,000 or more employees",
        ],
        "DevType": [
            "Developer, back-end;DevOps specialist",
            "Site Reliability Engineer",
            "Developer, full-stack",
            "Engineer, data",
            None,                           # missing — should be handled
        ],
        "Country": ["UK", "US", "Germany", "India", "Canada"],
    })


# ---------------------------------------------------------------------------
# load_data()
# ---------------------------------------------------------------------------

class TestLoadData:

    def test_returns_a_dataframe(self, tmp_path, raw_survey_df):
        """load_data() must return a pandas DataFrame."""
        csv_path = tmp_path / "survey_results_public.csv"
        raw_survey_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))

        assert isinstance(result, pd.DataFrame)

    def test_dataframe_is_not_empty(self, tmp_path, raw_survey_df):
        """Loaded DataFrame must contain at least one row."""
        csv_path = tmp_path / "survey_results_public.csv"
        raw_survey_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))

        assert len(result) > 0

    def test_expected_columns_are_present(self, tmp_path, raw_survey_df):
        """
        load_data() must preserve the core columns required for analysis.
        """
        csv_path = tmp_path / "survey_results_public.csv"
        raw_survey_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))

        required_columns = {"ResponseId", "CICDTools", "OrgSize", "DevType", "Country"}
        assert required_columns.issubset(set(result.columns))

    def test_raises_file_not_found_for_missing_path(self):
        """load_data() must raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_data("data/does_not_exist.csv")

    def test_response_id_is_unique(self, tmp_path, raw_survey_df):
        """Each row must have a unique ResponseId — no duplicates on load."""
        csv_path = tmp_path / "survey_results_public.csv"
        raw_survey_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))

        assert result["ResponseId"].is_unique


# ---------------------------------------------------------------------------
# clean_data()
# ---------------------------------------------------------------------------

class TestCleanData:

    def test_rows_with_null_cicdtools_are_dropped(self, loaded_df):
        """
        Rows where CICDTools is null cannot contribute to tool analysis
        and must be removed.
        """
        result = clean_data(loaded_df)

        assert result["CICDTools"].isnull().sum() == 0

    def test_row_count_reduced_after_dropping_nulls(self, loaded_df):
        """
        The cleaned DataFrame must have fewer rows than the raw one
        because at least one CICDTools value is null in the fixture.
        """
        result = clean_data(loaded_df)

        assert len(result) < len(loaded_df)

    def test_cicdtools_split_into_list(self, loaded_df):
        """
        CICDTools contains semicolon-separated strings in the raw data.
        clean_data() must split these into Python lists for analysis.
        e.g. "GitHub Actions;Jenkins" -> ["GitHub Actions", "Jenkins"]
        """
        result = clean_data(loaded_df)

        assert all(isinstance(val, list) for val in result["CICDTools"])

    def test_cicdtools_single_value_becomes_single_item_list(self, loaded_df):
        """
        A single tool with no semicolons should still become a list
        with one element, not a bare string.
        """
        result = clean_data(loaded_df)

        # Row with "GitLab CI/CD" has no semicolon — must still be a list
        gitlab_rows = result[
            result["CICDTools"].apply(lambda x: "GitLab CI/CD" in x)
        ]
        assert len(gitlab_rows) > 0
        assert all(isinstance(v, list) for v in gitlab_rows["CICDTools"])

    def test_cicdtools_values_are_stripped_of_whitespace(self, loaded_df):
        """
        Tool names must have leading/trailing whitespace stripped after
        splitting, to prevent duplicate entries like 'Jenkins' vs ' Jenkins'.
        """
        df = loaded_df.copy()
        df.loc[0, "CICDTools"] = "GitHub Actions ; Jenkins"   # spaces around semicolon

        result = clean_data(df)

        tools_in_first_row = result[result["ResponseId"] == 1]["CICDTools"].iloc[0]
        assert all(t == t.strip() for t in tools_in_first_row)

    def test_original_dataframe_is_not_mutated(self, loaded_df):
        """
        clean_data() must return a new DataFrame and not modify the
        original — standard defensive practice.
        """
        original_cicd = loaded_df["CICDTools"].copy()

        clean_data(loaded_df)

        pd.testing.assert_series_equal(loaded_df["CICDTools"], original_cicd)

    def test_response_id_column_is_preserved(self, loaded_df):
        """
        ResponseId must survive cleaning so rows remain traceable.
        """
        result = clean_data(loaded_df)

        assert "ResponseId" in result.columns

    def test_org_size_column_is_preserved(self, loaded_df):
        """
        OrgSize must survive cleaning — it is required for adoption-by-size
        analysis even though some values may be null.
        """
        result = clean_data(loaded_df)

        assert "OrgSize" in result.columns

    def test_dev_type_column_is_preserved(self, loaded_df):
        """
        DevType must survive cleaning — it is required for adoption-by-role
        analysis.
        """
        result = clean_data(loaded_df)

        assert "DevType" in result.columns

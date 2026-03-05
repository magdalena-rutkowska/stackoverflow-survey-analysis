"""
Tests for src/visualisation.py

Covers:
  - plot_tool_market_share()  : horizontal bar chart of CI/CD tool usage
  - plot_adoption_by_org_size(): grouped bar chart — tools vs company size
  - plot_adoption_by_devtype(): grouped bar chart — tools vs developer role

Tests verify that:
  - functions return matplotlib Figure objects (not None, not side-effect only)
  - output files are saved to disk when a path is provided
  - axes are labelled so charts are self-explanatory in submission
  - functions fail predictably on bad input

matplotlib is used in non-interactive mode throughout to prevent any
GUI windows from opening during test runs.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

# Force non-interactive backend before any other matplotlib import
matplotlib.use("Agg")

from src.visualisation import (
    plot_tool_market_share,
    plot_adoption_by_org_size,
    plot_adoption_by_devtype,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_plots():
    """
    Automatically close all matplotlib figures after every test to prevent
    resource warnings and state leaking between tests.
    """
    yield
    plt.close("all")


@pytest.fixture
def tool_counts():
    """Mirrors output of analysis.get_tool_counts()."""
    return pd.Series(
        {"GitHub Actions": 3, "Jenkins": 3, "CircleCI": 2, "GitLab CI/CD": 1},
        name="count",
    ).sort_values(ascending=False)


@pytest.fixture
def adoption_by_org_size():
    """Mirrors output of analysis.get_adoption_by_org_size()."""
    return pd.DataFrame(
        {
            "GitHub Actions": [2, 1, 0],
            "Jenkins":        [1, 1, 1],
            "CircleCI":       [1, 0, 1],
        },
        index=[
            "100 to 499 employees",
            "1,000 to 4,999 employees",
            "10,000 or more employees",
        ],
    )


@pytest.fixture
def adoption_by_devtype():
    """Mirrors output of analysis.get_adoption_by_devtype()."""
    return pd.DataFrame(
        {
            "GitHub Actions": [1, 2, 0],
            "Jenkins":        [2, 0, 1],
            "CircleCI":       [0, 1, 1],
        },
        index=[
            "DevOps specialist",
            "Site Reliability Engineer",
            "Developer, back-end",
        ],
    )


# ---------------------------------------------------------------------------
# plot_tool_market_share()
# ---------------------------------------------------------------------------

class TestPlotToolMarketShare:

    def test_returns_a_figure(self, tool_counts):
        """plot_tool_market_share() must return a matplotlib Figure."""
        result = plot_tool_market_share(tool_counts)

        assert isinstance(result, matplotlib.figure.Figure)

    def test_figure_has_one_axis(self, tool_counts):
        """The figure must contain exactly one Axes object."""
        result = plot_tool_market_share(tool_counts)

        assert len(result.axes) == 1

    def test_axis_has_title(self, tool_counts):
        """The chart must have a non-empty title for submission clarity."""
        result = plot_tool_market_share(tool_counts)

        title = result.axes[0].get_title()
        assert title is not None and len(title) > 0

    def test_axis_has_xlabel(self, tool_counts):
        """X-axis must be labelled so the unit of measurement is clear."""
        result = plot_tool_market_share(tool_counts)

        xlabel = result.axes[0].get_xlabel()
        assert xlabel is not None and len(xlabel) > 0

    def test_axis_has_ylabel(self, tool_counts):
        """Y-axis must be labelled so tool names are identified."""
        result = plot_tool_market_share(tool_counts)

        ylabel = result.axes[0].get_ylabel()
        assert ylabel is not None and len(ylabel) > 0

    def test_correct_number_of_bars(self, tool_counts):
        """Number of bars must equal the number of tools in the input."""
        result = plot_tool_market_share(tool_counts)
        ax = result.axes[0]

        # Count rendered bar patches
        bars = [p for p in ax.patches if p.get_width() > 0 or p.get_height() > 0]
        assert len(bars) == len(tool_counts)

    def test_saves_file_when_path_provided(self, tool_counts, tmp_path):
        """
        When an output path is supplied, the chart must be saved as a
        PNG file at that location.
        """
        output_path = str(tmp_path / "tool_market_share.png")

        plot_tool_market_share(tool_counts, output_path=output_path)

        assert os.path.exists(output_path)

    def test_saved_file_is_not_empty(self, tool_counts, tmp_path):
        """A saved PNG must have a non-zero file size."""
        output_path = str(tmp_path / "tool_market_share.png")

        plot_tool_market_share(tool_counts, output_path=output_path)

        assert os.path.getsize(output_path) > 0

    def test_raises_on_empty_series(self):
        """An empty Series must raise a ValueError — not produce a blank chart."""
        with pytest.raises(ValueError):
            plot_tool_market_share(pd.Series(dtype=int))


# ---------------------------------------------------------------------------
# plot_adoption_by_org_size()
# ---------------------------------------------------------------------------

class TestPlotAdoptionByOrgSize:

    def test_returns_a_figure(self, adoption_by_org_size):
        """plot_adoption_by_org_size() must return a matplotlib Figure."""
        result = plot_adoption_by_org_size(adoption_by_org_size)

        assert isinstance(result, matplotlib.figure.Figure)

    def test_figure_has_one_axis(self, adoption_by_org_size):
        """The figure must contain exactly one Axes object."""
        result = plot_adoption_by_org_size(adoption_by_org_size)

        assert len(result.axes) == 1

    def test_axis_has_title(self, adoption_by_org_size):
        """The chart must have a non-empty title."""
        result = plot_adoption_by_org_size(adoption_by_org_size)

        title = result.axes[0].get_title()
        assert title is not None and len(title) > 0

    def test_axis_has_xlabel(self, adoption_by_org_size):
        """X-axis must be labelled."""
        result = plot_adoption_by_org_size(adoption_by_org_size)

        xlabel = result.axes[0].get_xlabel()
        assert xlabel is not None and len(xlabel) > 0

    def test_axis_has_ylabel(self, adoption_by_org_size):
        """Y-axis must be labelled."""
        result = plot_adoption_by_org_size(adoption_by_org_size)

        ylabel = result.axes[0].get_ylabel()
        assert ylabel is not None and len(ylabel) > 0

    def test_legend_is_present(self, adoption_by_org_size):
        """
        A legend must be rendered so individual tools can be identified
        without reading axis tick labels.
        """
        result = plot_adoption_by_org_size(adoption_by_org_size)
        ax = result.axes[0]

        assert ax.get_legend() is not None

    def test_saves_file_when_path_provided(self, adoption_by_org_size, tmp_path):
        """Chart must be saved when an output path is supplied."""
        output_path = str(tmp_path / "adoption_by_org_size.png")

        plot_adoption_by_org_size(adoption_by_org_size, output_path=output_path)

        assert os.path.exists(output_path)

    def test_raises_on_empty_dataframe(self):
        """An empty DataFrame must raise a ValueError."""
        with pytest.raises(ValueError):
            plot_adoption_by_org_size(pd.DataFrame())


# ---------------------------------------------------------------------------
# plot_adoption_by_devtype()
# ---------------------------------------------------------------------------

class TestPlotAdoptionByDevtype:

    def test_returns_a_figure(self, adoption_by_devtype):
        """plot_adoption_by_devtype() must return a matplotlib Figure."""
        result = plot_adoption_by_devtype(adoption_by_devtype)

        assert isinstance(result, matplotlib.figure.Figure)

    def test_figure_has_one_axis(self, adoption_by_devtype):
        """The figure must contain exactly one Axes object."""
        result = plot_adoption_by_devtype(adoption_by_devtype)

        assert len(result.axes) == 1

    def test_axis_has_title(self, adoption_by_devtype):
        """The chart must have a non-empty title."""
        result = plot_adoption_by_devtype(adoption_by_devtype)

        title = result.axes[0].get_title()
        assert title is not None and len(title) > 0

    def test_axis_has_xlabel(self, adoption_by_devtype):
        """X-axis must be labelled."""
        result = plot_adoption_by_devtype(adoption_by_devtype)

        xlabel = result.axes[0].get_xlabel()
        assert xlabel is not None and len(xlabel) > 0

    def test_axis_has_ylabel(self, adoption_by_devtype):
        """Y-axis must be labelled."""
        result = plot_adoption_by_devtype(adoption_by_devtype)

        ylabel = result.axes[0].get_ylabel()
        assert ylabel is not None and len(ylabel) > 0

    def test_legend_is_present(self, adoption_by_devtype):
        """A legend must be rendered to identify individual tools."""
        result = plot_adoption_by_devtype(adoption_by_devtype)
        ax = result.axes[0]

        assert ax.get_legend() is not None

    def test_saves_file_when_path_provided(self, adoption_by_devtype, tmp_path):
        """Chart must be saved when an output path is supplied."""
        output_path = str(tmp_path / "adoption_by_devtype.png")

        plot_adoption_by_devtype(adoption_by_devtype, output_path=output_path)

        assert os.path.exists(output_path)

    def test_raises_on_empty_dataframe(self):
        """An empty DataFrame must raise a ValueError."""
        with pytest.raises(ValueError):
            plot_adoption_by_devtype(pd.DataFrame())

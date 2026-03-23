"""
visualisation.py

Responsibilities:
  - plot_tool_market_share()   : horizontal bar chart of overall CI/CD tool usage
  - plot_adoption_by_org_size(): grouped bar chart — tools vs company size
  - plot_adoption_by_devtype() : grouped bar chart — tools vs developer role

All functions:
  - Return a matplotlib Figure object
  - Accept an optional output_path to save the chart as a PNG
  - Raise ValueError on empty input rather than producing a blank chart
  - Use a consistent, clean visual style suitable for academic submission
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Use non-interactive backend — prevents GUI windows during test runs
# and when executed in headless environments
matplotlib.use("Agg")

# Consistent colour palette across all charts
PALETTE = [
    "#2196F3",  # blue
    "#FF9800",  # orange
    "#4CAF50",  # green
    "#E91E63",  # pink
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
]

FIGURE_SIZE = (12, 6)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 11


def _save_if_path(fig: matplotlib.figure.Figure, output_path) -> None:
    """Save figure to disk if a path was provided."""
    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)


def plot_tool_market_share(
    tool_counts: pd.Series,
    output_path=None,
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart showing how many respondents reported using
    each CI/CD tool.

    Parameters
    ----------
    tool_counts : pd.Series
        Tool names as index, integer counts as values.
        Expected to be sorted descending (as returned by get_tool_counts()).
    output_path : str, optional
        If provided, save the chart as a PNG at this path.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If tool_counts is empty.
    """
    if tool_counts.empty:
        raise ValueError("tool_counts is empty — nothing to plot.")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bars = ax.barh(
        tool_counts.index[::-1],   # reverse so highest count is at top
        tool_counts.values[::-1],
        color=PALETTE[0],
        edgecolor="white",
        linewidth=0.6,
    )

    # Add count labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (tool_counts.max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(
        "CI/CD Tool Usage — Stack Overflow Developer Survey 2024",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Number of Respondents", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("CI/CD Tool", fontsize=LABEL_FONTSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_if_path(fig, output_path)

    return fig


def plot_adoption_by_org_size(
    adoption_df: pd.DataFrame,
    output_path=None,
    top_n_tools: int = 6,
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart comparing CI/CD tool adoption across company sizes.

    Parameters
    ----------
    adoption_df : pd.DataFrame
        Index: OrgSize categories.
        Columns: CI/CD tool names.
        As returned by analysis.get_adoption_by_org_size().
    output_path : str, optional
        If provided, save the chart as a PNG at this path.
    top_n_tools : int
        Limit columns to the N most-used tools for readability.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If adoption_df is empty.
    """
    if adoption_df.empty:
        raise ValueError("adoption_df is empty — nothing to plot.")

    # Keep only the most-used tools to avoid an unreadable chart
    df = adoption_df[adoption_df.sum().nlargest(top_n_tools).index]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    df.plot(
        kind="bar",
        ax=ax,
        color=PALETTE[:len(df.columns)],
        edgecolor="white",
        linewidth=0.6,
        width=0.75,
    )

    ax.set_title(
        "CI/CD Tool Adoption by Organisation Size — Stack Overflow Developer Survey 2024",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Organisation Size", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Number of Respondents", fontsize=LABEL_FONTSIZE)
    ax.legend(
        title="CI/CD Tool",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right", fontsize=9)

    plt.tight_layout()
    _save_if_path(fig, output_path)

    return fig


def plot_adoption_by_devtype(
    adoption_df: pd.DataFrame,
    output_path=None,
    top_n_tools: int = 6,
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart comparing CI/CD tool adoption across developer roles.

    Parameters
    ----------
    adoption_df : pd.DataFrame
        Index: DevType categories.
        Columns: CI/CD tool names.
        As returned by analysis.get_adoption_by_devtype().
    output_path : str, optional
        If provided, save the chart as a PNG at this path.
    top_n_tools : int
        Limit columns to the N most-used tools for readability.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If adoption_df is empty.
    """
    if adoption_df.empty:
        raise ValueError("adoption_df is empty — nothing to plot.")

    # Limit to most-used tools
    df = adoption_df[adoption_df.sum().nlargest(top_n_tools).index]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    df.plot(
        kind="bar",
        ax=ax,
        color=PALETTE[:len(df.columns)],
        edgecolor="white",
        linewidth=0.6,
        width=0.75,
    )

    ax.set_title(
        "CI/CD Tool Adoption by Developer Role — Stack Overflow Developer Survey 2024",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Developer Role", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Number of Respondents", fontsize=LABEL_FONTSIZE)
    ax.legend(
        title="CI/CD Tool",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right", fontsize=9)

    plt.tight_layout()
    _save_if_path(fig, output_path)

    return fig

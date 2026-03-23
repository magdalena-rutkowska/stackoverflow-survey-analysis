"""
main.py

Entry point for the Stack Overflow Developer Survey CI/CD analysis.

Run with:
    python main.py

Expects the survey CSV at:
    data/survey_results_public.csv

Download from:
    https://survey.stackoverflow.co/  ->  "Download Full Data Set (CSV) 2024"

Outputs three PNG charts to the output/ directory.
"""

import os

from src.loader import load_data, clean_data
from src.analysis import (
    get_tool_counts,
    get_adoption_by_org_size,
    get_adoption_by_devtype,
    get_top_tools,
)
from src.visualisation import (
    plot_tool_market_share,
    plot_adoption_by_org_size,
    plot_adoption_by_devtype,
)

DATA_PATH = "data/stack-overflow-developer-survey-2024/survey_results_public.csv"
OUTPUT_DIR = "output"


def main():
    # ------------------------------------------------------------------
    # 1. Load and clean
    # ------------------------------------------------------------------
    print("Loading data...")
    raw_df = load_data(DATA_PATH)
    print(f"  Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")

    print("Cleaning data...")
    df = clean_data(raw_df)
    print(f"  {len(df):,} rows retained after removing null CICDTools responses")

    # ------------------------------------------------------------------
    # 2. Analysis — print summary statistics to console
    # ------------------------------------------------------------------
    print("\n--- Top 10 CI/CD Tools ---")
    top_tools = get_top_tools(df, n=10)
    for tool, count in top_tools.items():
        print(f"  {tool:<30} {count:>6,}")

    print("\n--- Adoption by Organisation Size (top 5 tools) ---")
    by_org = get_adoption_by_org_size(df)
    top_5_tools = get_tool_counts(df).head(5).index
    print(by_org[top_5_tools].to_string())

    print("\n--- Adoption by Developer Role (top 5 tools) ---")
    by_dev = get_adoption_by_devtype(df)
    print(by_dev[top_5_tools].to_string())

    # ------------------------------------------------------------------
    # 3. Visualisations — saved to output/
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nGenerating charts...")

    # Chart 1: Overall tool market share
    tool_counts = get_tool_counts(df)
    plot_tool_market_share(
        tool_counts,
        output_path=os.path.join(OUTPUT_DIR, "01_tool_market_share.png"),
    )
    print("  Saved: output/01_tool_market_share.png")

    # Chart 2: Adoption by organisation size
    plot_adoption_by_org_size(
        by_org,
        output_path=os.path.join(OUTPUT_DIR, "02_adoption_by_org_size.png"),
    )
    print("  Saved: output/02_adoption_by_org_size.png")

    # Chart 3: Adoption by developer role
    plot_adoption_by_devtype(
        by_dev,
        output_path=os.path.join(OUTPUT_DIR, "03_adoption_by_devtype.png"),
    )
    print("  Saved: output/03_adoption_by_devtype.png")

    print("\nDone. Charts saved to output/")


if __name__ == "__main__":
    main()

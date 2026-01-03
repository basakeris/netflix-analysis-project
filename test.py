import os
import pandas as pd
import pytest

# PATHS
RAW_FILE = "netflix_titles.csv"
CLEANED_FILE = "netflix_cleaned.csv"

CHART_FILES = [
    "chart_1_growth.png",
    "chart_2_countries.png",
    "chart_3_genres.png",
    "chart_4_types.png",
    "chart_5_ratings.png",
    "chart_6_heatmap.png",
    "chart_7_months.png",
    "chart_8_actors.png",
]

# DATA LOADING TESTS
def test_raw_data_exists():
    """Raw dataset should exist."""
    assert os.path.exists(RAW_FILE), "Raw Netflix dataset not found."


def test_cleaned_data_exists():
    """Cleaned dataset should be generated."""
    assert os.path.exists(CLEANED_FILE), "Cleaned dataset not found."


def test_cleaned_data_loadable():
    """Cleaned dataset should be readable."""
    df = pd.read_csv(CLEANED_FILE)
    assert not df.empty, "Cleaned dataset is empty."

# DATA CLEANING TESTS
def test_required_columns_exist():
    """Key columns created during cleaning should exist."""
    df = pd.read_csv(CLEANED_FILE)

    required_columns = [
        "date_added",
        "year_added",
        "month_added",
        "duration_numeric"
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nulls_in_critical_columns():
    """Critical columns should not contain null values."""
    df = pd.read_csv(CLEANED_FILE)

    critical_columns = ["date_added", "rating"]

    for col in critical_columns:
        assert df[col].isnull().sum() == 0, f"Null values found in {col}"

# ANALYSIS DATA VALIDATION
def test_content_type_distribution():
    """Dataset should contain Movies or TV Shows."""
    df = pd.read_csv(CLEANED_FILE)
    assert df["type"].nunique() >= 2, "Content type distribution seems incorrect."


def test_release_year_reasonable():
    """Release years should be within a reasonable range."""
    df = pd.read_csv(CLEANED_FILE)
    min_year = df["release_year"].min()
    max_year = df["release_year"].max()

    assert min_year >= 1900, "Unrealistic release year detected."
    assert max_year <= 2030, "Unrealistic future release year detected."


def test_country_data_present():
    """Country information should exist."""
    df = pd.read_csv(CLEANED_FILE)
    assert "country" in df.columns
    assert df["country"].notnull().sum() > 0

# VISUALIZATION OUTPUT TESTS
def test_charts_generated():
    """All visualization output files should exist."""
    missing = [f for f in CHART_FILES if not os.path.exists(f)]
    assert not missing, f"Missing chart files: {missing}"


def test_chart_files_not_empty():
    """Chart files should not be empty."""
    for chart in CHART_FILES:
        assert os.path.getsize(chart) > 0, f"{chart} is empty."

# NOTEBOOK / REPORT CONSISTENCY
def test_notebook_exists():
    """Final report notebook should exist."""
    assert os.path.exists("Final_Project_Report.ipynb"), "Final notebook not found."

# BASIC DATA QUALITY TEST
def test_dataset_size_reasonable():
    """Dataset should contain a reasonable number of records."""
    df = pd.read_csv(CLEANED_FILE)
    assert len(df) > 1000, "Dataset size is suspiciously small."

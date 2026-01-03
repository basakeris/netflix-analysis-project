"""
Data cleaning module for the Netflix dataset.
It handles missing values, fixes date formats, and converts durations to numeric values.
"""
import os
import pandas as pd


def load_data(csv_path):
    """
    Loads the raw Netflix dataset from the specified CSV path.
    """
    print("--- 1. Loading Data ---")
    try:
        # Dosya yolunu kontrol et
        if not os.path.exists(csv_path):
            print(f"ERROR: The file '{csv_path}' was not found in this folder.")
            print("Please make sure 'netflix_titles.csv' is in the same folder as this script.")
            return None

        df = pd.read_csv(csv_path)
        print(f"Dataset successfully loaded. Shape: {df.shape}")
        return df
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def clean_missing_values(df):
    """
    Handles missing values (NaN) by filling them with placeholders
    or dropping rows where necessary.
    """
    print("\n--- 2. Handling Missing Values ---")

    df = df.copy()

    # 1. Fill missing Director and Cast information
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')

    # 2. Fill missing Country information
    df['country'] = df['country'].fillna('Unknown Country')

    # 3. Drop rows where 'date_added' or 'rating' is missing
    before_drop = len(df)
    df = df.dropna(subset=['date_added', 'rating'])
    after_drop = len(df)

    print(f"Dropped {before_drop - after_drop} rows due to missing Date or Rating.")

    return df

def process_dates(df):
    """
    Converts the 'date_added' column to datetime objects.
    """
    print("\n--- 3. Processing Dates ---")

    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'])

    # Extract Year and Month
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month_name()

    print("Dates converted. Added 'year_added' and 'month_added' columns.")
    return df

def process_duration(df):
    """
    Converts duration strings (e.g., '90 min') into numeric values.
    """
    print("\n--- 4. Processing Duration ---")

    df['duration_numeric'] = df['duration'].copy()

    # Extract number if 'min' exists, else None
    df['duration_numeric'] = df['duration_numeric'].apply(
        lambda x: int(x.split(' ')[0]) if isinstance(x, str) and 'min' in x else None
    )

    print("Converted movie durations to numeric format.")
    return df

def run_cleaning_pipeline(raw_csv_path, output_path):
    """
    Executes the cleaning pipeline.
    """
    # 1. Load
    df = load_data(raw_csv_path)

    if df is not None:
        # 2. Clean
        df = clean_missing_values(df)

        # 3. Dates
        df = process_dates(df)

        # 4. Duration
        df = process_duration(df)

        # 5. Save
        df.to_csv(output_path, index=False)
        print(f"\n✅ SUCCESS! Cleaned data saved to: {output_path}")
        print(df.head())

if __name__ == "__main__":
    # AYNI KLASÖRDE OLDUKLARI İÇİN SADECE İSİM YAZIYORUZ
    RAW_FILE = "netflix_titles.csv"
    OUTPUT_FILE = "netflix_cleaned.csv"

    print(f"Looking for {RAW_FILE} in current folder...")
    run_cleaning_pipeline(RAW_FILE, OUTPUT_FILE)

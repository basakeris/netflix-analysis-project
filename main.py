"""
Main Execution Script for Netflix Data Analysis Project.
"""

import sys
import os
import subprocess

try:
    import data_cleaning
    import advanced_analysis
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import required modules. {e}")
    print("Please ensure 'data_cleaning.py' and 'advanced_analysis.py' are in the same directory.")
    sys.exit(1)


def main():
    print("="*70)
    print("NETFLIX DATA ANALYSIS PIPELINE - MAIN CONTROL")
    print("="*70)

    raw_data_path = 'netflix_titles.csv'
    cleaned_data_path = 'netflix_cleaned.csv'

    print("\nInitiating Data Cleaning Process...")

    if os.path.exists(raw_data_path):
        try:

            data_cleaning.run_cleaning_pipeline(
                raw_data_path, cleaned_data_path)
            print("✓ Data cleaning completed successfully.")
        except Exception as e:
            print(f"ERROR during data cleaning: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Raw data file '{raw_data_path}' not found!")
        print("Please place the dataset in the correct folder.")
        sys.exit(1)

    print("\nPerforming Statistical Analysis...")

    try:
        advanced_analysis.main()
        print("✓ Analysis completed successfully.")
    except AttributeError:
        print("WARNING: 'main()' function not found in advanced_analysis.py.")
    except Exception as e:
        print(f"ERROR during analysis: {e}")

    print("\nGenerating Visualizations...")

    # Since visualizations.py runs code at the global scope (not inside a function),
    visualization_script = "visualizations.py"

    if os.path.exists(visualization_script):
        try:
            subprocess.run([sys.executable, visualization_script], check=True)
            print("✓ Visualizations generated and saved.")
        except subprocess.CalledProcessError as e:
            print(
                f"ERROR: Visualization script failed with return code {e.returncode}.")
        except Exception as e:
            print(
                f"ERROR: An unexpected error occurred while running visualizations: {e}")
    else:
        print(f"ERROR: '{visualization_script}' file not found.")

    print("\n" + "="*70)
    print("PIPELINE EXECUTION FINISHED")
    print(f"Cleaned Data: {cleaned_data_path}")
    print("Check the project folder for generated charts (.png files).")
    print("="*70)


if __name__ == "__main__":
    main()

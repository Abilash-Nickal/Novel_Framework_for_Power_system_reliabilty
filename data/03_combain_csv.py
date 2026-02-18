import csv
from typing import List, Dict, Any, Tuple

# Define the input and output file names
INPUT_FILES = [
    "03.gen_sigle_column_MW.csv",
    "03.gen_sigle_column_cost.csv"

]
OUTPUT_FILE = "CEB_GEN_cost_for_each_unit.csv"


def read_csv_data(filename: str) -> Tuple[List[str], List[List[Any]]]:
    """Reads a CSV file and returns its header and data rows."""
    header: List[str] = []
    data: List[List[Any]] = []

    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            # Read header
            try:
                header = next(reader)
            except StopIteration:
                pass  # File is empty

            # Read data
            data = [row for row in reader]

        print(f"Read {len(data)} rows from {filename}.")
        return header, data
    except FileNotFoundError:
        print(f"Warning: File not found: {filename}. Skipping this file.")
        return [], []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], []


def combine_and_export_csvs(input_files: List[str], output_file: str):
    """Combines data from multiple CSV files side-by-side."""

    all_data_columns: List[List[List[Any]]] = []
    all_headers: List[str] = []
    max_rows = 0

    # --- 1. Read all input files ---
    for filename in input_files:
        header, data = read_csv_data(filename)

        # Determine the effective columns for this file and name them
        # MODIFIED: Removed the filename from the header for cleaner output
        file_headers = header
        all_headers.extend(file_headers)

        all_data_columns.append(data)
        if len(data) > max_rows:
            max_rows = len(data)

    if max_rows == 0:
        print("No data found in any input files to combine.")
        return

    print(f"Combining data into {max_rows} total rows...")

    # --- 2. Create the unified rows (padding shorter files with empty cells) ---
    unified_rows: List[List[Any]] = []

    for i in range(max_rows):
        current_unified_row: List[Any] = []

        # Iterate through each file's data
        for file_data in all_data_columns:
            # Check if this file has a row at the current index (i)
            if i < len(file_data):
                # If yes, extend the row with the full content of the file's row
                current_unified_row.extend(file_data[i])
            else:
                # If no (file is shorter), extend the row with empty strings
                # to match the width (number of columns) of that file's header.
                file_width = len(file_data[0]) if file_data else 0
                current_unified_row.extend([''] * file_width)

        unified_rows.append(current_unified_row)

    # --- 3. Export the combined data ---
    print(f"Exporting combined data to '{output_file}'...")
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the unified header
            writer.writerow(all_headers)

            # Write all unified data rows
            writer.writerows(unified_rows)

        print(f"Successfully exported combined data to {output_file}")
    except Exception as e:
        print(f"An error occurred during CSV export: {e}")


# --- Main Execution ---
combine_and_export_csvs(INPUT_FILES, OUTPUT_FILE)
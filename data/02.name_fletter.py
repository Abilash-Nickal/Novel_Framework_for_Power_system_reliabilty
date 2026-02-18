import csv
from typing import List, Any
from datetime import datetime

INPUT_FILENAME = "02.gen_sep_name.csv"
OUTPUT_FILENAME = "03.gen_single_column_NAMES.csv"  # Renamed output for clarity


def is_number(s):
    """Helper function to check if a string represents a valid number (non-empty and convertible to float)."""
    # Check for empty or blank string
    if not s or s.strip() == '':
        return False
    try:
        # Check if it can be converted to float (i.e., it's a unit capacity/FOR value)
        float(s)
        return True
    except ValueError:
        return False


def flatten_csv_to_single_column_names(input_file: str = INPUT_FILENAME, output_file: str = OUTPUT_FILENAME):
    """
    Reads a wide CSV file, determines the repetition count (number of unit columns)
    for each row, and exports the corresponding name/string into a single, long column.
    """
    all_names_repeated: List[str] = []

    # --- 1. Read Data ---
    print(f"Reading wide data from '{input_file}'...")
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            # Skip the header row
            try:
                next(reader)
            except StopIteration:
                print(f"Error: '{input_file}' is empty or has no header.")
                return

            valid_rows = 0
            for row in reader:
                if not row:
                    continue

                # The Name is in the first column (index 0)
                name = row[0].strip()

                # The values start from the second column (index 1 onwards)
                repetition_count = 0
                # Iterate from the second column onward
                for item in row[1:]:
                    if is_number(item):
                        repetition_count += 1
                    elif item.strip() == '':
                        # Stop counting units if we hit an empty string (end of repetition block)
                        break
                    else:
                        # If we hit text that isn't the name (which should be col 0)
                        break  # Stop on any unexpected text

                # Repeat the name by the count
                if repetition_count > 0 and name:
                    all_names_repeated.extend([name] * repetition_count)
                    valid_rows += 1

        if not all_names_repeated:
            print(f"Successfully processed {valid_rows} rows, but found 0 names to flatten.")
            return

        print(f"Successfully generated {len(all_names_repeated)} total name entries.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please ensure it exists.")
        return

    # --- 2. Export Data ---
    print(f"Exporting flattened name data to '{output_file}'...")
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the single column header
            writer.writerow(['Unit Name'])

            # Write each item in the list as a new row
            writer.writerows([[item] for item in all_names_repeated])

        print(f"Successfully exported flattened name data to {output_file}")
    except Exception as e:
        print(f"An error occurred during CSV export: {e}")
        return


# --- Main Execution ---
flatten_csv_to_single_column_names()
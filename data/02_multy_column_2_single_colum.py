import csv
from typing import List, Any

INPUT_FILENAME = "02.gen_sep_name.csv"  # Using the wide output as the example input
OUTPUT_FILENAME = "03.gen_sigle_column_name.csv"


def flatten_csv_to_single_column(input_file: str = INPUT_FILENAME, output_file: str = OUTPUT_FILENAME):
    """
    Reads all columns and rows from an input CSV, extracts all numerical data,
    and exports it to a new CSV file with a single, flattened column.
    """
    all_numbers: List[str] = []

    # --- 1. Read Data ---
    print(f"Reading all columns from '{input_file}'...")
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
                for item in row:
                    try:
                        # Attempt to convert the item to a float
                        value = float(item.strip())
                        # Store the valid number as a string
                        all_numbers.append(str(value))
                    except ValueError:
                        # Ignore empty strings or non-numerical text
                        continue
                valid_rows += 1

        if not all_numbers:
            print(f"Successfully read {valid_rows} rows, but found 0 valid numbers.")
            return

        print(f"Successfully extracted {len(all_numbers)} total numbers from {valid_rows} rows.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please ensure it exists.")
        return

    # --- 2. Export Data ---
    print(f"Exporting flattened data to '{output_file}'...")
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the single column header
            writer.writerow(['All Numbers'])

            # Write each item in the list as a new row
            writer.writerows([[item] for item in all_numbers])

        print(f"Successfully exported flattened data to {output_file}")
    except Exception as e:
        print(f"An error occurred during CSV export: {e}")
        return


# --- Main Execution ---
flatten_csv_to_single_column()
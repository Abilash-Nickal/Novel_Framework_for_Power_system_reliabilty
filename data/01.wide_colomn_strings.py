import csv
from typing import List, Tuple, Any

INPUT_FILENAME = "01.GEN_names_&_units.csv"
OUTPUT_FILENAME = "02.gen_sep_name.csv"


def import_from_csv(filename: str = INPUT_FILENAME) -> List[Tuple[int, Any]]:
    """
    Reads data from the specified CSV file.
    Expects two columns:
    Col 1 (repetition count, int),
    Col 2 (Value/Name to Repeat, Any).
    """
    data = []
    print(f"Reading data from '{filename}'...")
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            # Skip the header row
            try:
                next(reader)
            except StopIteration:
                print(f"Error: '{filename}' is empty.")
                return []

            for row in reader:
                # Check for exactly 2 columns
                if len(row) >= 2:
                    try:
                        # Col 1: Repetition count (INT)
                        col1 = int(row[0].strip())

                        # Col 2: Value/Name to be repeated (STRING)
                        col2 = row[1].strip()

                        # Ensure count is valid
                        if col1 < 0:
                            print(f"Skipping row with invalid count (<0): {row}")
                            continue

                        data.append((col1, col2))
                    except ValueError:
                        print(f"Skipping row with invalid repetition count (not integer): {row}")
                        continue
                else:
                    print(f"Skipping malformed row (less than 2 columns): {row}")
        print(f"Successfully read {len(data)} valid rows.")
        return data
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        return []


def process_data_wide(data: List[Tuple[int, Any]]) -> Tuple[List[List[Any]], int]:
    """
    Processes the raw data: Output is Col 2 (Value/Name), followed by Col 2 repeated Col 1 times.
    Returns the processed data and the maximum number of repeated value columns required.
    """
    processed_rows = []
    max_cols = 0

    if not data:
        return processed_rows, 0

    # 1. Determine the maximum required column count (the largest Col 1 value)
    max_cols = max(col1 for col1, _ in data)
    print(f"Maximum repetition count found: {max_cols}. Output CSV will have {max_cols + 1} total data columns.")

    # 2. Generate the output rows
    for col1, col2 in data:

        # Start the row with the Value/Name itself (since it's also the unique identifier here)
        output_row = [col2]

        if col1 > 0:
            # Add the repeated values (Col 2)
            # The output columns will contain the repeated name/value.
            output_row.extend([col2] * col1)

        # Pad the rest of the row with empty strings to match max_cols
        # Note: We need to pad (max_cols - col1) spaces to the *right* of the repeated values.
        padding = max_cols - col1
        output_row.extend([''] * padding)

        # The first column (Name) is already present, so the total length must be max_cols + 1.

        processed_rows.append(output_row)

    return processed_rows, max_cols


def export_to_csv(data_rows: List[List[Any]], max_cols: int, filename: str = OUTPUT_FILENAME):
    """
    Exports the processed data to a CSV file with dynamic column headers.
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Generate headers: 'Name/Value' + ('Value 1', 'Value 2', ...)
            headers = ['Unit Name/Value'] + [f'Unit {i + 1}' for i in range(max_cols)]
            writer.writerow(headers)

            # Write the data rows
            writer.writerows(data_rows)

        print(f"Successfully exported wide data to {filename}")
    except Exception as e:
        print(f"An error occurred during CSV export: {e}")


# --- Main Execution ---
raw_data = import_from_csv()

if raw_data:
    processed_data, max_columns = process_data_wide(raw_data)
    export_to_csv(processed_data, max_columns)

    # Console Preview
    print(f"\n--- Console Preview (Unit Name + {max_columns} Unit Columns) ---")
    headers = ['Name'] + [f'U{i + 1}' for i in range(max_columns)]
    print("| " + " | ".join(headers) + " |")
    print("---" * (max_columns + 2))
    for row in processed_data[:5]:  # Show first 5 rows
        # Ensure row is printed correctly, even with embedded spaces in Name
        # Only print the first max_columns + 1 elements
        print("| " + " | ".join(map(str, row[:max_columns + 1])) + " |")
    if len(processed_data) > 5:
        print("...")
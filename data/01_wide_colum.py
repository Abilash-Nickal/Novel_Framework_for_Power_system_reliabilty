import csv
from typing import List, Tuple, Any

INPUT_FILENAME = "01.GEN_names_&_units.csv"
OUTPUT_FILENAME = "02.gen_sep_name.csv"


def import_from_csv(filename: str = INPUT_FILENAME) -> List[Tuple[int, float]]:
    """
    Reads data from the specified CSV file.
    Expects two columns: Col 1 (repetition count, int), Col 2 (value, float).
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
                if len(row) == 2:
                    try:
                        # Convert Col 1 to int (repetition count)
                        col1 = int(row[0].strip())
                        # Convert Col 2 to float (value to be repeated)
                        col2 = float(row[1].strip())
                        data.append((col1, col2))
                    except ValueError:
                        print(f"Skipping row with non-numerical data: {row}")
                        continue
                else:
                    print(f"Skipping malformed row: {row}")
        print(f"Successfully read {len(data)} rows.")
        return data
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        return []


def process_data_wide(data: List[Tuple[int, float]]) -> Tuple[List[List[Any]], int]:
    """
    Processes the raw data: Output is Col 2 repeated Col 1 times into separate columns.
    Returns the processed data and the maximum number of columns required.
    """
    processed_rows = []
    max_cols = 0

    if not data:
        return processed_rows, 0

    # 1. Determine the maximum required column count (the largest Col 1 value)
    max_cols = max(col1 for col1, _ in data)
    print(f"Maximum repetition count found: {max_cols}. Output CSV will have {max_cols} value columns.")

    # 2. Generate the output rows
    for col1, col2 in data:
        output_row = []
        if col1 > 0:
            # Add the repeated values
            output_row.extend([str(col2)] * col1)

        # Pad the rest of the row with empty strings to match max_cols
        padding = max_cols - len(output_row)
        output_row.extend([''] * padding)

        processed_rows.append(output_row)

    return processed_rows, max_cols


def export_to_csv(data_rows: List[List[Any]], max_cols: int, filename: str = OUTPUT_FILENAME):
    """
    Exports the processed data to a CSV file with dynamic column headers.
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Generate headers (e.g., 'Value 1', 'Value 2', 'Value 3')
            headers = [f'Value {i + 1}' for i in range(max_cols)]
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
    print(f"\n--- Console Preview ({max_columns} Columns) ---")
    print("| " + " | ".join([f'V{i + 1}' for i in range(max_columns)]) + " |")
    print("---" * (max_columns + 1))
    for row in processed_data[:5]:  # Show first 5 rows
        print("| " + " | ".join(row) + " |")
    if len(processed_data) > 5:
        print("...")
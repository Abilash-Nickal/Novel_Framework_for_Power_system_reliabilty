import csv
import os
import math


def generate_annual_load_profile(input_filename, output_filename, total_hours=8760):
    """
    Reads a 24-hour load profile from an input CSV and repeats it to create an 8760-hour annual profile.

    Args:
        input_filename (str): The name of the file containing the 24 hourly loads.
        output_filename (str): The name of the file to save the 8760 hourly loads.
        total_hours (int): The target total hours for the year (default 8760).
    """
    daily_loads = []

    # 1. Read the 24-hour load profile
    try:
        with open(input_filename, 'r', newline='') as infile:
            reader = csv.reader(infile)

            # Skip the header row (assuming the first row is 'Load')
            header = next(reader, None)
            if header is None:
                print(f"Error: Input file {input_filename} is empty.")
                return

            # Read remaining rows
            for row in reader:
                if row and row[0].strip():
                    try:
                        # Convert the load value to a float and store it
                        daily_loads.append(float(row[0].strip()))
                    except ValueError:
                        print(f"Warning: Skipping non-numeric value: {row[0]}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return

    if not daily_loads:
        print("Error: Could not read any valid load data from the input file.")
        return

    num_daily_points = len(daily_loads)
    if num_daily_points != 24:
        print(f"Warning: Found {num_daily_points} data points, not 24. Repeating the available data.")

    # Calculate the number of times the daily profile must be repeated
    num_days = math.ceil(total_hours / num_daily_points)

    # 2. Generate the 8760-hour sequence
    annual_loads = []

    # Repeat the daily load profile for the required number of days
    for day in range(num_days):
        # Extend the list by appending the daily_loads sequence
        annual_loads.extend(daily_loads)

    # Trim the list to exactly 8760 hours
    final_annual_loads = annual_loads[:total_hours]

    # 3. Write the 8760 hourly data to a new CSV file
    try:
        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            # Write the header
            writer.writerow(['Load (MW)'])

            # Write the 8760 hours of data
            for i, load in enumerate(final_annual_loads):
                # Hour starts at 1
                writer.writerow([load])

        print(f"\n✅ Successfully generated and saved annual load profile ({total_hours} hours) to '{output_filename}'")
        print(f"File saved in the current project directory: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"❌ An error occurred while writing the output file: {e}")


# --- Execution ---
INPUT_FILE = "Lanka_L.csv"
OUTPUT_FILE = "SriLanka_Load_8760hr_repeat.csv"

# Run the function
generate_annual_load_profile(INPUT_FILE, OUTPUT_FILE)
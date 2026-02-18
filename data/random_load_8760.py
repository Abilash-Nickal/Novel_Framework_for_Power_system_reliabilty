import csv
import os
import math
import random  # Import the random module


def generate_annual_load_profile(input_filename, output_filename, total_hours=8760, deviation_range=0.01):
    """
    Reads a 24-hour load profile from an input CSV and repeats it to create an 8760-hour annual profile,
    adding a random deviation to each hourly load value.

    Args:
        input_filename (str): The name of the file containing the 24 hourly loads.
        output_filename (str): The name of the file to save the 8760 hourly loads.
        total_hours (int): The target total hours for the year (default 8760).
        deviation_range (float): The load will deviate by +/- this percentage (e.g., 0.01 means +/- 1%).
                                 The original request asked for a factor between 1 and 2, but for load data
                                 this usually means a small percentage change applied to the original value.
                                 Here we apply a deviation between 1 and 2% (1.0 to 2.0% change).
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

    # 2. Generate the 8760-hour sequence with random deviation
    final_annual_loads = []

    # Repeat the daily load profile for the required number of days
    for day in range(num_days):
        for hour_load in daily_loads:
            # Check if the total hours target has been reached
            if len(final_annual_loads) >= total_hours:
                break

            # --- Apply Random Deviation (1 to 2% difference from base) ---
            # Generate a random factor between 0.01 and 0.02 (1% to 2%)
            # We want the factor to shift the load, so we use 1 +/- factor.
            # R is a random number between -0.01 and +0.02 (1% below to 2% above).
            deviation_factor = random.uniform(0.01, 0.02)

            # Randomly make it subtractive 50% of the time, or additive 50% of the time.
            if random.choice([True, False]):
                # Factor is 1 - (1% to 2% of load) -> load decreases
                final_factor = 1.0 - deviation_factor
            else:
                # Factor is 1 + (1% to 2% of load) -> load increases
                final_factor = 1.0 + deviation_factor

            # New load is the base load multiplied by the random factor
            new_load = hour_load * final_factor
            final_annual_loads.append(new_load)

    # 3. Write the 8760 hourly data to a new CSV file
    try:
        with open(output_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            # Write the header
            writer.writerow(['Load (MW)'])

            # Write the 8760 hours of data
            for i, load in enumerate(final_annual_loads):
                # Hour starts at 1
                writer.writerow([i +1,load])

        print(f"\n✅ Successfully generated and saved annual load profile ({total_hours} hours) to '{output_filename}'")
        print(f"File saved in the current project directory: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"❌ An error occurred while writing the output file: {e}")


# --- Execution ---
INPUT_FILE = "Lanka_L.csv"
OUTPUT_FILE = "SriLanka_Load_8760hr_Random.csv"  # Changed output name for clarity

# Run the function
# We will apply a random factor that deviates the load between -2% and +2%
generate_annual_load_profile(INPUT_FILE, OUTPUT_FILE)
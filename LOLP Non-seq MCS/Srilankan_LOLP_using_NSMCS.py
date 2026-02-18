import random
import csv
import os
import math
import numpy as np  # Add numpy for simulation processing
import pandas as pd  # Add pandas for efficient data loading
from datetime import datetime

# --- Configuration ---
NUM_ITERATIONS = 87600000 # Number of Monte Carlo samples
# --- UPDATING FILE NAME FOR GENERATOR DATA ---
GEN_DATA_FILE = "../data/CEB_GEN_FOR_for_each_unit.csv"

LOAD_DATA_FILE = "../data/SriLanka_Load_8760hr_repeat.csv"

# --- Global Data Storage ---
Gen = []  # Generator Capacities (MW)
FOR = []  # Forced Outage Rates
Annual_Load_Profile = np.array([])  # Use NumPy array for fast sampling
method = 'NSMCS'

# --- 1. Data Loading Functions ---

# Gen data load
def load_generator_data(filepath):
    """Loads Gen and FOR data from CSV using Pandas."""
    global Gen, FOR
    Gen = []
    FOR = []

    try:
        # Use pandas to load generator data. Assumes header exists.
        # Loads columns 0 (Capacity) and 1 (FOR)
        # ASSUMPTION: The first column is Capacity (MW) and the second is FOR.
        df_gen = pd.read_csv(filepath, header=0, usecols=[0, 1])

        # Extract columns and convert to standard Python lists
        Gen = df_gen.iloc[:, 0].astype(float).tolist()
        FOR = df_gen.iloc[:, 1].astype(float).tolist()

    except FileNotFoundError:
        print(f"ERROR: Generator data file '{filepath}' not found.")
        return False
    except Exception as e:
        print(f"ERROR loading generator data: {e}")
        return False

    print(f" Loaded {len(Gen)} generators.")
    return True

# Annual load data lode
def load_annual_load_profile(filepath):
    """Loads the 8760 load values from CSV using Pandas."""
    global Annual_Load_Profile

    try:
        # Load the entire CSV.
        df_load = pd.read_csv(filepath, header=0)

        # We assume the load data is in the FIRST column (index 0) of the data frame.
        Annual_Load_Profile = df_load.iloc[:, 0].astype(float).values

    except FileNotFoundError:
        print(f"ERROR: Load data file '{filepath}' not found.")
        print("Please ensure 'SriLanka_Load_8760hr_Random.csv' is in your project directory.")
        return False
    except Exception as e:
        print(f"ERROR loading load data: {e}")
        return False

    print(f" Loaded {len(Annual_Load_Profile)} hourly load values.")
    return True


# --- 2. Monte Carlo Simulation ---

def run_monte_carlo():
    """Performs the Non-Sequential Monte Carlo Simulation."""
    if not Gen or Annual_Load_Profile.size == 0:
        print("\nFATAL: Data not loaded. Exiting simulation.")
        return

    H = 0  # Loss of Load Events
    N = 0  # Total Iterations
    num_generators = len(Gen)
    num_load_hours = Annual_Load_Profile.size  # Use NumPy array size

    # Convert Gen and FOR lists to NumPy arrays for faster computation within the loop
    Gen_np = np.array(Gen)
    FOR_np = np.array(FOR)

    print(f"\n--- Starting Monte Carlo Simulation ({NUM_ITERATIONS:,} iterations) ---")

    # Pre-generate all random numbers for efficiency
    random_gen_checks = np.random.random((NUM_ITERATIONS, num_generators))
    random_load_indices = np.random.randint(0, num_load_hours, NUM_ITERATIONS)

    # Use standard loop for clarity
    for n in range(NUM_ITERATIONS):
        N += 1

        # A. Check Generator Availability (Vectorized Summation)
        # 1. Create a boolean mask: True if random check > FOR (i.e., generator is UP)
        outage_mask = random_gen_checks[n] > FOR_np
        # 2. Sum the capacities only where the mask is True (Generator * 1)
        availableGen = np.sum(Gen_np * outage_mask)

        # B. Select Current Load (Random Sample from 8760 Profile)
        load_index = random_load_indices[n]
        currentLoad = Annual_Load_Profile[load_index]

        # C. Check for Loss of Load
        if currentLoad > availableGen:
            H += 1

        # Optional: Log progress every 1 million iterations
        if N % 1000000 == 0:
            print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")

    # --- Results ---
    LOLP = H / N
    LOLE = LOLP * 8760
    # STORE DATA
    # 1. Define the file name and the header/data
    CSV_FILE_NAME = '../reliability_results.csv'
    HEADER = ['Timestamp','Method', 'H_Events', 'N_Iterations', 'LOLP', 'LOLE_Hours']
    data_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'NSMSC',
        H,
        N,
        LOLP,
        LOLE
    ]

    # 2. Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(CSV_FILE_NAME)

    # 3. Open the file in append ('a') mode
    with open(CSV_FILE_NAME, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file does not exist, write the header row first
        if not file_exists:
            writer.writerow(HEADER)
            print(f" Created new file: {CSV_FILE_NAME} and wrote header.")

        # Write the data row (this happens every time the script is run)
        writer.writerow(data_row)

    print(f" Results appended to {CSV_FILE_NAME}.")
    print("\n--- Final Results ---")
    print(f"Total Loss of Load Events (H): {H:,}")
    print(f"Total Iterations (N): {N:,}")
    print(f"Loss of Load Probability (LOLP): {LOLP:.10f}")
    print(f"Loss of Load Expectation (LOLE): {LOLE:.2f} hours/year")


# --- 3. Execution Flow ---

if __name__ == "__main__":
    # Remove placeholder creation logic as the user now has a specific file name

    # Attempt to load all data
    # Note: Pandas must be installed for this to work (pip install pandas numpy)
    gen_success = load_generator_data(GEN_DATA_FILE)
    load_success = load_annual_load_profile(LOAD_DATA_FILE)

    # Run simulation if data loaded successfully
    if gen_success and load_success:
        run_monte_carlo()
    else:
        print("\nSimulation aborted due to missing or invalid input files. Check console output for specific file errors.")



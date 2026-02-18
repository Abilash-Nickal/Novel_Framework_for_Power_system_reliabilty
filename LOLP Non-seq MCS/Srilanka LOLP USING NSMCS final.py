import random
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- Configuration ---
NUM_ITERATIONS = 100000
HOURS_PER_YEAR = 8760

# --- File Paths ---
# Assuming the file path is correct relative to the script location
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"


# --- 1. Data Loading Functions ---

def load_generator_data(filepath):
    """
    Loads Gen (Capacity) and FOR data from CSV using Pandas by column name.
    Assumes columns are named 'GEN' and 'FOR' and ignores the first column (Unit Name).
    """

    if not os.path.exists(filepath):
        print(f"ERROR: Generator data file '{filepath}' not found.")
        return [], []

    try:
        # Load ONLY the columns explicitly named
        df_gen = pd.read_csv(filepath, header=0, usecols=['Unit Capacity (MW)', 'Unit FOR'])

        # Load Capacity (GEN) and FOR lists using the specified names
        Gen_list = df_gen['Unit Capacity (MW)'].astype(float).tolist()
        FOR_series = df_gen['Unit FOR'].astype(float)  # Load as Series to apply validation

        # --- VALIDATION AND CAPTURE ---
        # 1. Ensure FOR values are non-negative
        FOR_series = FOR_series.clip(lower=0.0)

        # 2. Ensure FOR values do not exceed 1.0 (100% probability)
        FOR_series = FOR_series.clip(upper=1.0)

        FOR_list = FOR_series.tolist()

    except Exception as e:
        print(f"ERROR loading generator data: {e}. Check if column names 'Unit Capacity (MW)' and 'Unit FOR' exist.")
        return [], []

    print(f" Loaded {len(Gen_list)} generators.")
    return Gen_list, FOR_list


def load_annual_load_profile(filepath):
    """Loads 8760 hourly load profile from CSV using Pandas/NumPy."""

    if not os.path.exists(filepath):
        print(f"ERROR: Load data file '{filepath}' not found.")
        return np.array([])

    try:
        df_load = pd.read_csv(filepath, header=0)
        # Use .iloc[:, 0] to reliably get the first data column (Load) after the header
        Annual_Load_Profile_np = df_load.iloc[:, 0].astype(float).values

    except Exception as e:
        print(f"ERROR loading load data: {e}")
        return np.array([])

    print(f" Loaded {len(Annual_Load_Profile_np)} hourly load values.")
    return Annual_Load_Profile_np


# --- 2. Monte Carlo Simulation (Memory Efficient) ---

def run_monte_carlo(Gen_list, FOR_list, Annual_Load_Profile_np, run_label):
    """
    Performs the Non-Sequential Monte Carlo Simulation, generating random numbers
    inside the loop to save memory.
    """
    if not Gen_list or Annual_Load_Profile_np.size == 0:
        print("\nFATAL: Data not loaded. Exiting simulation.")
        return 0, 0
    H = 0
    N = 0
    num_generators = len(Gen_list)
    num_load_hours = Annual_Load_Profile_np.size

    Gen_np = np.array(Gen_list)
    FOR_np = np.array(FOR_list)

    print(f"\n--- Starting Monte Carlo Simulation: {run_label} ({NUM_ITERATIONS:,} iter) ---")
    start_time = datetime.now()

    # --- CORE LOOP: Memory Efficient ---
    for n in range(NUM_ITERATIONS):

        N += 1

        # A. Check Generator Availability (Memory-Efficient: Generate N random numbers here)
        random_gen_checks_n = np.random.random(num_generators)

        # 1. Create a boolean mask: True if random check > FOR (i.e., generator is UP)
        outage_mask = random_gen_checks_n > FOR_np

        # 2. Sum the capacities only where the mask is True
        availableGen = np.sum(Gen_np * outage_mask)

        # B. Select Current Load
        load_index = np.random.randint(0, num_load_hours)
        currentLoad = Annual_Load_Profile_np[load_index]

        # C. Check for Loss of Load
        if currentLoad > availableGen:
            H += 1

        if N % 1000000 == 0:
            print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Results ---
    LOLP = H / N
    LOLE = LOLP * HOURS_PER_YEAR

    print("\n--- Final Results ---")
    print(f"Simulation Duration: {duration}")
    print(f"Final LOLP (Probability): {LOLP:.8f}")
    print(f"Final LOLE (Expected Hours/Year): {LOLE:.2f}")

    return LOLP, LOLE


# --- 3. Result Saving Function ---

def save_lolp_results(run_label, total_iterations, lolp, lole, method, output_file='../reliability_results.csv'):


    import csv
    import os

    header = ['Timestamp', 'Method', 'Run_Label', 'Total_Iterations', 'LOLP', 'LOLE_Hrs_Yr']
    data_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        method,
        run_label,
        total_iterations,
        f"{lolp:.10f}",  # Ensure 10 decimal precision for LOLP in CSV
        f"{lole:.2f}"
    ]

    file_exists = os.path.exists(output_file)

    try:
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if the file did not exist before this run
            if not file_exists:
                writer.writerow(header)

            # Write the data row
            writer.writerow(data_row)

        print(f" Results appended to {output_file}")

    except Exception as e:
        print(f" Error saving results: {e}")


# --- 4. Execution Flow (The Two-Run Comparison) ---

if __name__ == "__main__":

    # 1. Load Data Globally
    # Assumes your CSV has columns named 'GEN' and 'FOR'
    Gen, FOR = load_generator_data(GEN_DATA_FILE)
    Annual_Load_Profile = load_annual_load_profile(LOAD_DATA_FILE)

    if Gen and Annual_Load_Profile.size > 0:

        # --- RUN 1: Base Case ---
        run_label_1 = "Base_System_FOR"
        print(f"\n==== Starting Run 1: {run_label_1} ====")

        lolp_1, lole_1 = run_monte_carlo(Gen, FOR, Annual_Load_Profile, run_label_1)
        save_lolp_results(run_label_1, NUM_ITERATIONS, lolp_1, lole_1, method='NSMCS')

    else:
        print(
            "\nSimulation aborted due to missing or invalid input files. Check console output for specific file errors.")
import random
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
NUM_ITERATIONS = 10000000
HOURS_PER_YEAR = 8760

# --- File Paths ---
GEN_DATA_FILE = "../data/CEB_GEN_FOR_for_each_unit.csv"
LOAD_DATA_FILE = "../data/SriLanka_Load_8760hr_repeat.csv"


# --- 1. Data Loading Functions ---

def load_generator_data(filepath):
    """Loads Gen and FOR data from CSV using Pandas and validates FOR is between 0 and 1."""

    if not os.path.exists(filepath):
        print(f"ERROR: Generator data file '{filepath}' not found.")
        return [], []

    try:
        # Load columns 0 (Capacity) and 1 (FOR)
        df_gen = pd.read_csv(filepath, header=0, usecols=[0, 1])

        Gen_list = df_gen.iloc[:, 0].astype(float).tolist()
        FOR_series = df_gen.iloc[:, 1].astype(float)  # Load as Series to apply validation

        # --- VALIDATION AND CAPTURE ---
        # 1. Ensure FOR values are non-negative
        FOR_series = FOR_series.clip(lower=0.0)

        # 2. Ensure FOR values do not exceed 1.0 (100% probability)
        # If a value is > 1 (e.g., 1.1), it is set to 1.0
        FOR_series = FOR_series.clip(upper=1.0)

        # Convert the cleaned Series back to a list
        FOR_list = FOR_series.tolist()

        # Check for any extreme values that were capped
        if (FOR_series > 1.0).any() or (FOR_series < 0.0).any():
            print("WARNING: Some FOR values were outside the [0, 1] range and have been capped.")

    except Exception as e:
        print(f"ERROR loading generator data: {e}")
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

    # --- CORE LOOP: Memory Fix Applied ---
    for n in range(NUM_ITERATIONS):
        N += 1

        # A. Check Generator Availability (Memory-Efficient: Generate N random numbers here)
        # IMPORTANT: np.random.random() already guarantees numbers are between 0.0 and 1.0,
        # which is why this check works correctly against the validated FOR_np array.
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
    print(f"LOLP events : {H}")
    print(f"Simulation Duration: {duration}")
    print(f"Final LOLP (Probability): {LOLP:.8f}")
    print(f"Final LOLE (Expected Hours/Year): {LOLE:.2f}")

    return LOLP, LOLE


# --- 3. Result Saving Function ---

def save_lolp_results(run_label, total_iterations, lolp, lole, method, output_file='../reliability_results.csv'):
    """Appends simulation results (LOLP, LOLE) to a CSV file."""

    import csv
    import os

    header = ['Timestamp', 'Method', 'Run_Label', 'Total_Iterations', 'LOLP', 'LOLE_Hrs_Yr']
    data_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        method,
        run_label,
        total_iterations,
        lolp,
        lole
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

        print(f"Results appended to {output_file}")

    except Exception as e:
        print(f"Error saving results: {e}")


# --- 4. Execution Flow (The Two-Run Comparison) ---

if __name__ == "__main__":

    # 1. Load Data Globally
    Gen, FOR = load_generator_data(GEN_DATA_FILE)
    Annual_Load_Profile = load_annual_load_profile(LOAD_DATA_FILE)

    if Gen and Annual_Load_Profile.size > 0:

        # --- RUN 1: Base Case ---
        run_label_1 = "Base_System_FOR"
        print(f"\n==== Starting Run 1: {run_label_1} ====")

        # Pass data explicitly to run_monte_carlo
        lolp_1, lole_1 = run_monte_carlo(Gen, FOR, Annual_Load_Profile, run_label_1)
        save_lolp_results(run_label_1, NUM_ITERATIONS, lolp_1, lole_1, method='NSMCS')

        # --- RUN 2: Modified Case (e.g., FOR increased by 10%) ---

        # Create a new FOR list where all original FORs are increased by 10%
        # The validation in load_generator_data ensures the base FORs are <= 1.0.
        FOR_case_2 = [min(1.0, f * 1.10) for f in FOR]
        run_label_2 = "FOR_Increase_10pct"

        print(f"\n==== Starting Run 2: {run_label_2} ====")

        # Pass the modified FOR list
        lolp_2, lole_2 = run_monte_carlo(Gen, FOR_case_2, Annual_Load_Profile, run_label_2)
        save_lolp_results(run_label_2, NUM_ITERATIONS, lolp_2, lole_2, method='NSMCS')

    else:
        print(
            "\nSimulation aborted due to missing or invalid input files. Check console output for specific file errors.")
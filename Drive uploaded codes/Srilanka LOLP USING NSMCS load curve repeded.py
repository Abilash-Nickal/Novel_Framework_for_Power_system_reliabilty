import random
import csv
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
NUM_ITERATIONS = 10000000  # Number of Monte Carlo samples
HOURS_PER_YEAR = 8760  # Standard number of hours in a non-leap year

# --- File Paths ---
GEN_DATA_FILE = "CEB_GEN_FOR_for_each_unit.csv"
LOAD_DATA_FILE = "SriLanka_Load_8760hr_repeat.csv"


# --- 1. Data Loading Functions ---
def load_generator_data(filepath):
    """Loads Gen and FOR data from CSV using Pandas."""

    if not os.path.exists(filepath):
        print(f"ERROR: Generator data file '{filepath}' not found.")
        return [], []

    try:
        # Load columns 0 (Capacity) and 1 (FOR)
        df_gen = pd.read_csv(filepath, header=0, usecols=[0, 1])

        # Extract columns and convert to standard Python lists
        Gen_list = df_gen.iloc[:, 0].astype(float).tolist()
        FOR_list = df_gen.iloc[:, 1].astype(float).tolist()

    except Exception as e:
        print(f"ERROR loading generator data: {e}")
        return [], []

    print(f"Loaded {len(Gen_list)} generators.")
    return Gen_list, FOR_list


def load_annual_load_profile(filepath):
    """Loads 8760 hourly load profile from CSV using Pandas/NumPy."""

    if not os.path.exists(filepath):
        print(f"ERROR: Load data file '{filepath}' not found.")
        return np.array([])

    try:
        # Load the entire CSV. Assuming the load data is in the first data column.
        df_load = pd.read_csv(filepath, header=0)

        # Use .iloc[:, 0] to reliably get the first data column after the header
        Annual_Load_Profile_np = df_load.iloc[:, 0].astype(float).values

    except Exception as e:
        print(f"ERROR loading load data: {e}")
        return np.array([])

    print(f"Loaded {len(Annual_Load_Profile_np)} hourly load values.")
    return Annual_Load_Profile_np


# --- 2. Monte Carlo Simulation (Refactored for efficiency) ---

def run_monte_carlo(Gen_list, FOR_list, Annual_Load_Profile_np, run_label):
    """Performs the Non-Sequential Monte Carlo Simulation."""
    if not Gen_list or Annual_Load_Profile_np.size == 0:
        print("\nFATAL: Data not loaded. Exiting simulation.")
        return 0, 0

    H = 0  # Loss of Load Events (Total Failure Hours)
    N = 0  # Total Iterations
    num_generators = len(Gen_list)
    num_load_hours = Annual_Load_Profile_np.size

    # Convert lists to NumPy arrays for vectorized operations
    Gen_np = np.array(Gen_list)
    FOR_np = np.array(FOR_list)

    print(f"\n--- Starting Monte Carlo Simulation: {run_label} ({NUM_ITERATIONS:,} iter) ---")
    start_time = datetime.now()

    # Use standard loop for clarity, GENERATING RANDOM NUMBERS INSIDE THE LOOP
    for n in range(NUM_ITERATIONS):
        N += 1

        # A. Check Generator Availability (In-Loop Generation to save RAM)
        random_gen_checks_n = np.random.random(num_generators)

        # 1. Create a boolean mask: True if random check > FOR (i.e., generator is UP)
        outage_mask = random_gen_checks_n > FOR_np

        # 2. Sum the capacities only where the mask is True (Generator * 1)
        availableGen = np.sum(Gen_np * outage_mask)

        # B. Select Current Load (Random Sample from 8760 Profile)
        load_index = np.random.randint(0, num_load_hours)
        currentLoad = Annual_Load_Profile_np[load_index]

        # C. Check for Loss of Load
        if currentLoad > availableGen:
            H += 1

        # Optional: Log progress every 1 million iterations
        if N % 1000000 == 0:
            print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Results ---
    LOLP = H / N
    LOLE = LOLP * HOURS_PER_YEAR  # LOLE = LOLP * 8760

    print("\n--- Final Results ---")
    print(f"Simulation Duration: {duration}")
    print(f"Total Loss of Load Events (H): {H:,}")
    print(f"Total Iterations (N): {N:,}")
    print(f"Final LOLP (Probability): {LOLP:.10f}")
    print(f"Final LOLE (Expected Hours/Year): {LOLE:.2f}")

    return LOLP, LOLE  # RETURN RESULTS

# --- 4. Execution Flow  ---

if __name__ == "__main__":

    # 1. Load Data Globally (only done once)
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
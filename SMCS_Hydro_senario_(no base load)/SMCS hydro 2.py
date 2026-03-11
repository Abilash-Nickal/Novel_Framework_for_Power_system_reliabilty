import numpy as np
import pandas as pd
import os
import random  # Included for completeness, though np.random.rand() is used
from datetime import datetime  # Required for timing the simulation

# --- Configuration ---
# Set to 100,000 years for statistical accuracy and practical runtime
NUM_YEARS = 1000
# The number of hours in a non-leap year
HOURS_PER_YEAR = 8760

# monthly hydro capacity cap (MW) for each month (Jan..Dec).
MONTHLY_HYDRO_CAPACITY_MW = [1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500]


# --- File Names ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"


# --- 1. Data Loading Functions ---

def load_data():
    """Loads all necessary data from CSV files."""
    try:
        # 1. Load Generator Data (Capacity, MTTF, MTTR)
        df_gen = pd.read_csv(GEN_DATA_FILE, header=0)

        # Ensure correct columns exist and are numeric
        Gen_df = df_gen[['Unit Capacity (MW)', 'MTTR (hours)', 'MTTF (hours)']]
        Gen_df = Gen_df.astype(float)

        Gen = Gen_df['Unit Capacity (MW)'].values
        MTTF = Gen_df['MTTF (hours)'].values
        MTTR = Gen_df['MTTR (hours)'].values

        # Identify hydro units
        plant_types = df_gen['TYPES'].astype(str).str.upper()
        is_hydro = (plant_types == 'HYDRO').values

        print(f" Loaded {len(Gen)} generators. Hydro generators: {is_hydro.sum()} | thermal generators: {(~is_hydro).sum()}")

        # 2. Load Chronological Load Data
        df_load = pd.read_csv(LOAD_DATA_FILE, header=0)

        # Assuming load is in the first data column (index 0)
        Annual_Load_Profile = df_load.iloc[:, 0].astype(float).values

        # Check if the load profile is valid for a whole year repetition
        if len(Annual_Load_Profile) < HOURS_PER_YEAR:
            # If the load file is short (e.g., 24 hours), repeat it to fill the year
            num_repetitions = int(np.ceil(HOURS_PER_YEAR / len(Annual_Load_Profile)))
            Annual_Load_Profile = np.tile(Annual_Load_Profile, num_repetitions)[:HOURS_PER_YEAR]
            print(f" Loaded and extended load profile to {len(Annual_Load_Profile)} hours.")
        else:
            # Otherwise, use the first 8760 hours
            Annual_Load_Profile = Annual_Load_Profile[:HOURS_PER_YEAR]
            print(f" Loaded {len(Annual_Load_Profile)} hourly load values.")

        return Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

# --- 2. Sequential Monte Carlo Simulation ---

def run_sequential_mcs(Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro, monthly_hydro_capacity_mw=None):
    """
    Runs the SMCS over multiple simulated years.
    Uses time-to-event sampling (exponential distribution).
    """
    num_generators = len(Gen)
    total_study_hours = NUM_YEARS * HOURS_PER_YEAR

    # Initialize state arrays
    # 0: Available (UP), 1: Unavailable (DOWN)
    state = np.zeros(num_generators, dtype=int)

    # Time until next change (failure if UP, repair if DOWN)
    time_to_next_event = np.zeros(num_generators)

    # Initialize Time to Failure for all units (as they start UP)
    # Using -MTTF * ln(rand) to sample from exponential distribution
    time_to_next_event = -MTTF * np.log(np.random.rand(num_generators))

    # Cumulative results
    total_LOL_hours = 0

    current_time = 0

    # Build month lookup per hour (0-11 for Jan-Dec)
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_by_hour = np.zeros(HOURS_PER_YEAR, dtype=int)
    h = 0
    for m, days in enumerate(month_lengths):
        hours_in_month = days * 24
        month_by_hour[h:h + hours_in_month] = m
        h += hours_in_month

    # Prepare monthly hydro capacity caps
    total_nominal_hydro = np.sum(Gen[is_hydro])
    if monthly_hydro_capacity_mw is None:
        monthly_caps = np.full(12, total_nominal_hydro)
    else:
        monthly_caps = np.array(monthly_hydro_capacity_mw, dtype=float)
        if monthly_caps.size != 12:
            raise ValueError("monthly_hydro_capacity_mw must have 12 values (Jan..Dec)")

    print(f" Using monthly hydro capacity caps (MW): {monthly_caps}")

    print(f"--- Starting Sequential Monte Carlo for {NUM_YEARS} years ({total_study_hours:,} hours) ---")

    # Start time for performance measurement
    start_time = datetime.now()

    # Main Simulation Loop
    while current_time < total_study_hours:

        # 1. Determine the Available Generation (Supply)
        # Separate hydro and non-hydro availability, then cap hydro to monthly rated value
        annual_hour = int(current_time) % HOURS_PER_YEAR
        month_idx = month_by_hour[annual_hour]
        available_hydro_raw = np.sum(Gen[(state == 0) & is_hydro])
        available_hydro_capped = min(available_hydro_raw, monthly_caps[month_idx])
        available_non_hydro = np.sum(Gen[(state == 0) & (~is_hydro)])
        availableGen = available_non_hydro + available_hydro_capped

        # 2. Determine the Load (Demand)
        # Use the chronological hour of the year (repeats yearly)
        currentLoad = Annual_Load_Profile[annual_hour]

        # 3. Determine the Time Step to Advance
        # Find the minimum time until the next event (failure or repair) across all units
        min_time_step = np.min(time_to_next_event)

        # Cap time step at 1 hour to ensure hourly load changes are met.
        time_step = min(1.0, min_time_step)

        # 4. Check for Loss of Load (LOL) and accumulate failure duration
        if availableGen < currentLoad:
            # We are in an outage state. Accumulate the duration of the failure.
            total_LOL_hours += time_step

            # 5. Advance Time and Update Generator States

        # a. Advance the system clock
        current_time += time_step

        # b. Decrease time to event for all generators
        time_to_next_event -= time_step

        # Log progress periodically (every 1000 simulated years)
        if current_time % (HOURS_PER_YEAR * 1000) <= time_step and current_time > 0:
            print(
                f"Progress: Year {int(current_time // HOURS_PER_YEAR)} completed. Total LOL Hours so far: {total_LOL_hours:,.2f}")

        # c. Check for completed events (where time_to_next_event <= 0)
        units_changing_state = np.where(time_to_next_event <= 1e-6)[0]  # Use a small tolerance for zero

        for i in units_changing_state:
            # Flip the state (0 -> 1 or 1 -> 0)
            state[i] = 1 - state[i]

            # Calculate new time to next event (TTF if now UP, TTR if now DOWN)
            if state[i] == 0:  # Unit just repaired, now UP -> next event is a FAILURE (uses MTTF)
                time_to_next_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:  # Unit just failed, now DOWN -> next event is REPAIR (uses MTTR)
                time_to_next_event[i] = -MTTR[i] * np.log(np.random.rand())

    # --- Results ---
    total_simulated_hours = total_study_hours
    LOLP = total_LOL_hours / total_simulated_hours
    LOLE = total_LOL_hours / NUM_YEARS  # Loss of Load Expectation (hours/year)

    # End time for performance measurement
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n--- Sequential Monte Carlo Final Results ---")
    print(f"Total Study Duration: {total_simulated_hours:,} hours ({NUM_YEARS} years)")
    print(f"Simulation Runtime: {duration}")
    print(f"Total Loss of Load Hours (H): {total_LOL_hours:,.2f}")
    print(f"Loss of Load Probability (LOLP): {LOLP:.8f}")
    print(f"Loss of Load Expectation (LOLE): {LOLE:.2f} hours/year")


# --- 3. Execution Flow ---

if __name__ == "__main__":
    # 1. Load data
    Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro = load_data()

    # 2. Run simulation
    if Gen is not None:
        run_sequential_mcs(
            Gen,
            MTTF,
            MTTR,
            Annual_Load_Profile,
            is_hydro,
            monthly_hydro_capacity_mw=MONTHLY_HYDRO_CAPACITY_MW,
        )
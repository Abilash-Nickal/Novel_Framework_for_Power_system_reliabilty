import numpy as np
import pandas as pd
import os
from datetime import datetime
import random


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(gen_file, load_file):
    try:
        df_gen = pd.read_csv(gen_file)

        # Validation for required columns
        required_gen_cols = ['Capacity (MW)', 'MTTF (hours)', 'MTTR (hours)']
        if not all(col in df_gen.columns for col in required_gen_cols):
            raise ValueError(f"Generator file is missing one or more required columns: {required_gen_cols}")

        Gen = df_gen['Capacity (MW)'].values
        MTTF = df_gen['MTTF (hours)'].values
        MTTR = df_gen['MTTR (hours)'].values

        df_load = pd.read_csv(load_file)
        Load = df_load.iloc[:, 0].values.astype(float)

        # Ensure 8760-hour profile
        if len(Load) < 8760:
            Load = np.tile(Load, int(np.ceil(8760 / len(Load))))[:8760]
        else:
            Load = Load[:8760]

        print(f"Data loaded successfully: {len(Gen)} generators, {len(Load)} load hours.")
        return Gen, MTTF, MTTR, Load

    except Exception as e:
        print(f"FATAL DATA LOADING ERROR: {e}")
        # Return None on failure
        return None, None, None, None


# -----------------------------
# 2. Sequential Monte Carlo Simulation
# -----------------------------
def run_smcs(Gen, MTTF, MTTR, Load, years=10000):
    HOURS = 8760
    total_hours = HOURS * years

    n = len(Gen)

    # All generators start UP (0 = UP, 1 = DOWN)
    state = np.zeros(n, dtype=int)

    # Initial TTF for all generators
    next_event = -MTTF * np.log(np.random.rand(n))

    LOL_hours = 0
    t = 0
    start_time = datetime.now()

    while t < total_hours:

        # Available capacity (sum of all UP units)
        available = np.sum(Gen[state == 0])

        # Current chronological hourly demand
        load_now = Load[int(t) % 8760]

        # Time until next failure/repair event, capped at 1 hour for load change
        dt = min(1.0, np.min(next_event))

        # If load > supply, add outage time
        if available < load_now:
            LOL_hours += dt

        # Advance time
        t += dt
        next_event -= dt

        # Handle generators whose event happened (failure or repair)
        # Use tolerance (1e-6) for checking zero
        events = np.where(next_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]  # flip state

            # Assign next event time
            if state[i] == 0:
                # Just repaired, next event is FAILURE (using MTTF)
                next_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                # Just failed, next event is REPAIR (using MTTR)
                next_event[i] = -MTTR[i] * np.log(np.random.rand())

        # Simple progress logger
        if int(t) % (HOURS * 1000) == 0 and t > 0 and dt < 1.0:  # Only print when clock catches up
            print(f"Progress: Year {int(t) // HOURS} completed. LOL Hours: {LOL_hours:,.2f}")

    # Results
    end_time = datetime.now()
    duration = end_time - start_time

    LOLP = LOL_hours / total_hours
    LOLE = LOL_hours / years

    print("\n----- SMCS RESULTS -----")
    print(f"Total Simulated Years: {years:,}")
    print(f"Simulation Runtime: {duration}")
    print(f"Total Loss of Load Hours: {LOL_hours:,.2f}")
    print(f"LOLP (Probability): {LOLP:.8f}")
    print(f"LOLE (Expected Hours/Year): {LOLE:.2f} hours/year")

    return LOLP, LOLE


# -----------------------------
# 3. Main
# -----------------------------
if __name__ == "__main__":

    # Define Scenario Parameters
    YEARS_TO_RUN = 10000
    GEN_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
    LOAD_FILE = "../data/SriLanka_Load_8760hr_repeat.csv"

    # Load Data
    Gen, MTTF, MTTR, Load = load_data(GEN_FILE, LOAD_FILE)

    if Gen is not None:
        # Run Simulation
        run_smcs(Gen, MTTF, MTTR, Load, years=YEARS_TO_RUN)
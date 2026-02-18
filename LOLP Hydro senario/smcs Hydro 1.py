import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
NUM_YEARS = 1000
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# Example: 12 values for Jan-Dec
HYDRO_MONTHLY_CAP = np.array([
    1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500
])


# =========================================================
# DATA LOADING (FIXED)
# =========================================================
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # --- FIX #1: CLEAN THE TEXT ---
    # Convert to uppercase and remove spaces so "Hydro " matches "HYDRO"
    GenType = df_gen['TYPES'].str.upper().str.strip().values

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # --- FIX #2: PRECISE CALENDAR ---
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days):
        month_lookup.extend([m] * (days * 24))
    month_lookup = np.array(month_lookup)

    return Gen, MTTF, MTTR, GenType, Annual_Load, month_lookup


# =========================================================
# SIMULATION LOOP
# =========================================================
def run_smcs(Gen, MTTF, MTTR, GenType, Load, month_lookup):
    n_gen = len(Gen)
    state = np.zeros(n_gen, dtype=int)
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    total_LOEE = 0.0  # Added back LOEE
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")
    start_clock = datetime.now()

    while current_time < TOTAL_HOURS:
        hour_of_year = int(current_time) % HOURS_PER_YEAR

        # --- FIX #2: USE LOOKUP ---
        month = month_lookup[hour_of_year]
        current_load = Load[hour_of_year]

        # Identify Available Generators
        up_mask = (state == 0)

        # --- FIX #1: ROBUST MATCHING ---
        hydro_mask = (GenType == 'HYDRO') & up_mask
        thermal_mask = (GenType == 'THERMAL') & up_mask

        available_hydro = np.sum(Gen[hydro_mask])
        available_thermal = np.sum(Gen[thermal_mask])

        # Apply Hydro Limit
        hydro_cap = HYDRO_MONTHLY_CAP[month]
        effective_hydro = min(available_hydro, hydro_cap)

        effective_generation = available_thermal + effective_hydro

        # Determine Time Step
        min_event_time = np.min(time_to_event)
        time_step = min(1.0, min_event_time)

        # Reliability Check
        if effective_generation < current_load:
            deficit = current_load - effective_generation
            total_LOL_hours += time_step
            total_LOEE += deficit * time_step  # Calculate Energy Missing

        # Advance Time
        current_time += time_step
        time_to_event -= time_step

        # Handle Events
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            if state[i] == 0:
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

    # Results
    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS
    LOEE = total_LOEE / NUM_YEARS

    print("\n========= FINAL RESULTS =========")
    print(f"LOLE (Hours/Year) : {LOLE:.4f}")
    print(f"LOEE (MWh/Year)   : {LOEE:.2f}")
    print(f"Simulation Time   : {datetime.now() - start_clock}")


if __name__ == "__main__":
    Gen, MTTF, MTTR, GenType, Load, month_lookup = load_data()
    run_smcs(Gen, MTTF, MTTR, GenType, Load, month_lookup)
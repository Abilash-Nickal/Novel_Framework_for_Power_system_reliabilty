import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- Configuration ---
NUM_YEARS = 3000
HOURS_PER_YEAR = 8760

# Monthly hydro capacity caps (MW) Jan..Dec
MONTHLY_HYDRO_CAPACITY_MW = [
    853, 866, 1011, 916, 1023, 1133,
    1061, 964, 939, 1057, 1184, 1118
]

# --- File Names ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    Gen = df_gen['Unit Capacity (MW)'].astype(float).values
    MTTF = df_gen['MTTF (hours)'].astype(float).values
    MTTR = df_gen['MTTR (hours)'].astype(float).values

    plant_types = df_gen['TYPES'].astype(str).str.upper()
    is_hydro = (plant_types == 'HYDRO').values

    print(f"Loaded {len(Gen)} generators")
    print(f"Hydro: {is_hydro.sum()} | Non-hydro: {(~is_hydro).sum()}")

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load_Profile = df_load.iloc[:, 0].astype(float).values

    if len(Annual_Load_Profile) < HOURS_PER_YEAR:
        Annual_Load_Profile = np.tile(
            Annual_Load_Profile,
            int(np.ceil(HOURS_PER_YEAR / len(Annual_Load_Profile)))
        )[:HOURS_PER_YEAR]

    return Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro

# ---------------------------------------------------------------------
# 2. SEQUENTIAL MONTE CARLO SIMULATION
# ---------------------------------------------------------------------
def run_sequential_mcs(
    Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro,
    monthly_hydro_capacity_mw
):

    num_generators = len(Gen)
    total_study_hours = NUM_YEARS * HOURS_PER_YEAR

    # Generator states: 0 = UP, 1 = DOWN
    state = np.zeros(num_generators, dtype=int)

    # Time to next event
    time_to_next_event = -MTTF * np.log(np.random.rand(num_generators))

    # -----------------------------
    # Reliability accumulators
    # -----------------------------
    total_LOL_hours = 0.0   # hours
    total_LOEE = 0.0        # MWh

    current_time = 0.0

    # Month lookup table
    month_lengths = [31,28,31,30,31,30,31,31,30,31,30,31]
    month_by_hour = np.zeros(HOURS_PER_YEAR, dtype=int)
    h = 0
    for m, d in enumerate(month_lengths):
        month_by_hour[h:h + d*24] = m
        h += d*24

    monthly_caps = np.array(monthly_hydro_capacity_mw, dtype=float)

    print("\n--- Starting Sequential Monte Carlo Simulation ---")
    print(f"Years simulated: {NUM_YEARS}")
    print(f"Monthly hydro caps (MW): {monthly_caps}")

    start_time = datetime.now()

    # -----------------------------------------------------------------
    # MAIN EVENT-DRIVEN LOOP
    # -----------------------------------------------------------------
    while current_time < total_study_hours:

        annual_hour = int(current_time) % HOURS_PER_YEAR
        month_idx = month_by_hour[annual_hour]

        # Available generation
        hydro_raw = np.sum(Gen[(state == 0) & is_hydro])
        hydro_capped = min(hydro_raw, monthly_caps[month_idx])
        non_hydro = np.sum(Gen[(state == 0) & (~is_hydro)])

        availableGen = hydro_capped + non_hydro
        currentLoad = Annual_Load_Profile[annual_hour]

        # Time step
        min_time_step = np.min(time_to_next_event)
        time_step = min(1.0, min_time_step)

        # -----------------------------
        # LOSS OF LOAD CHECK
        # -----------------------------
        if availableGen < currentLoad:
            ENS = currentLoad - availableGen   # MW

            total_LOL_hours += time_step
            total_LOEE += ENS * time_step      # MWh

        # Advance time
        current_time += time_step
        time_to_next_event -= time_step

        # Progress log
        if current_time % (HOURS_PER_YEAR * 100) <= time_step:
            print(f"Progress: {int(current_time//HOURS_PER_YEAR)} years")

        # Handle events
        changing_units = np.where(time_to_next_event <= 1e-9)[0]
        for i in changing_units:
            state[i] = 1 - state[i]

            if state[i] == 0:   # repaired → UP
                time_to_next_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:               # failed → DOWN
                time_to_next_event[i] = -MTTR[i] * np.log(np.random.rand())

    # -----------------------------------------------------------------
    # FINAL INDICES
    # -----------------------------------------------------------------
    LOLP = total_LOL_hours / total_study_hours
    LOLE = total_LOL_hours / NUM_YEARS
    LOEE = total_LOEE / NUM_YEARS

    duration = datetime.now() - start_time

    print("\n--- FINAL RESULTS ---")
    print(f"Simulation runtime: {duration}")
    print(f"LOLP  = {LOLP:.8f}")
    print(f"LOLE  = {LOLE:.2f} hours/year")
    print(f"LOEE  = {LOEE:,.2f} MWh/year")

# ---------------------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":

    Gen, MTTF, MTTR, Annual_Load_Profile, is_hydro = load_data()

    run_sequential_mcs(
        Gen,
        MTTF,
        MTTR,
        Annual_Load_Profile,
        is_hydro,
        MONTHLY_HYDRO_CAPACITY_MW
    )

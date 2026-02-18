import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================================================

NUM_YEARS=5000
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# =========================================================
# 2. INPUT FILES
# =========================================================

GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# =========================================================
# 3. MONTHLY HYDRO CAPACITY (MW) – SCENARIO INPUT
# =========================================================

HYDRO_MONTHLY_CAP = np.array(
   [853,866,1011,916,1023,1133,1061,964,939,1057,1184,1118])

# =========================================================
# 4. DATA LOADING
# =========================================================

def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    # Required columns:
    # Unit Capacity (MW), MTTF (hours), MTTR (hours), Type
    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # Generator type: THERMAL or HYDRO
    GenType = df_gen['TYPES'].values  # string array

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)


    return Gen, MTTF, MTTR, GenType, Annual_Load

# =========================================================
# 5. SEQUENTIAL MONTE CARLO SIMULATION
# =========================================================

def run_smcs(Gen, MTTF, MTTR, GenType, Load):

    n_gen = len(Gen)

    # Generator state: 0 = UP, 1 = DOWN
    state = np.zeros(n_gen, dtype=int)

    # Time to next event
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")

    start_clock = datetime.now()

    while current_time < TOTAL_HOURS:

        # ---------------------------------------------
        # Time indices
        # ---------------------------------------------
        hour_of_year = int(current_time) % HOURS_PER_YEAR
        month = min(hour_of_year // 730, 11)

        current_load = Load[hour_of_year]

        # ---------------------------------------------
        # AVAILABLE GENERATION
        # ---------------------------------------------
        up_mask = (state == 0)

        hydro_mask = (GenType == 'HYDRO') & up_mask
        thermal_mask = (GenType == 'THERMAL') & up_mask

        available_hydro = np.sum(Gen[hydro_mask])
        available_thermal = np.sum(Gen[thermal_mask])

        # ---------------------------------------------
        # MONTHLY HYDRO CAP APPLICATION
        # ---------------------------------------------
        hydro_cap = HYDRO_MONTHLY_CAP[month]
        effective_hydro = min(available_hydro, hydro_cap)

        effective_generation = available_thermal + effective_hydro

        # ---------------------------------------------
        # LOSS OF LOAD CHECK
        # ---------------------------------------------
        min_event_time = np.min(time_to_event)
        time_step = min(1.0, min_event_time)

        if effective_generation < current_load:
            total_LOL_hours += time_step

        # ---------------------------------------------
        # ADVANCE TIME
        # ---------------------------------------------
        current_time += time_step
        time_to_event -= time_step

        # ---------------------------------------------
        # HANDLE EVENTS (FAILURE / REPAIR)
        # ---------------------------------------------
        events = np.where(time_to_event <= 1e-6)[0]

        for i in events:
            state[i] = 1 - state[i]

            if state[i] == 0:
                # Repaired → sample time to failure
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                # Failed → sample repair time
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # ---------------------------------------------
        # PROGRESS LOG
        # ---------------------------------------------
        if int(current_time) % (HOURS_PER_YEAR * 1000) < time_step:
            print(f"Year {int(current_time // HOURS_PER_YEAR)} completed")

    # =================================================
    # RESULTS
    # =================================================

    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS

    runtime = datetime.now() - start_clock

    print("\n========= FINAL RESULTS =========")
    print(f"Total LOL Hours   : {total_LOL_hours:.2f}")
    print(f"LOLP              : {LOLP:.8f}")
    print(f"LOLE              : {LOLE:.2f} hours/year")
    print(f"Simulation Time   : {runtime}")

# =========================================================
# 6. MAIN
# =========================================================

if __name__ == "__main__":

    Gen, MTTF, MTTR, GenType, Load = load_data()
    run_smcs(Gen, MTTF, MTTR, GenType, Load)

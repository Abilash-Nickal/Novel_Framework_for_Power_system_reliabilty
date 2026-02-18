import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================================================

NUM_YEARS = 1000
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# =========================================================
# 2. INPUT FILES
# =========================================================

GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# =========================================================
# 3. MONTHLY HYDRO CAPACITY (MW)
# =========================================================

HYDRO_MONTHLY_CAP = np.array(
    [853,866,1011,916,1023,1133,1061,964,939,1057,1184,1118]
)

# =========================================================
# 4. DATA LOADING
# =========================================================

def load_data():

    df_gen = pd.read_csv(GEN_DATA_FILE)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)
    GenType = df_gen['TYPES'].values.astype(str)
    UnitCost = df_gen['Unit Cost (LKR/kWh)'].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    return Gen, MTTF, MTTR, GenType, UnitCost, Annual_Load


# =========================================================
# 5. SEQUENTIAL MONTE CARLO SIMULATION
# =========================================================

def run_smcs(Gen, MTTF, MTTR, GenType, UnitCost, Load):

    n_gen = len(Gen)

    # Generator state: 0 = UP, 1 = DOWN
    state = np.zeros(n_gen, dtype=int)

    # Initial time to failure
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")
    start_clock = datetime.now()

    while current_time < TOTAL_HOURS:

        # -------------------------------------------------
        # TIME INDEX
        # -------------------------------------------------
        hour_of_year = int(current_time) % HOURS_PER_YEAR
        month = min(hour_of_year // 730, 11)
        current_load = Load[hour_of_year]

        # -------------------------------------------------
        # FIND AVAILABLE GENERATORS
        # -------------------------------------------------
        up_indices = np.where(state == 0)[0]

        if len(up_indices) == 0:
            effective_generation = 0.0
        else:

            available_cap = Gen[up_indices]
            available_cost = UnitCost[up_indices]
            available_type = GenType[up_indices]

            # ---------------------------------------------
            # APPLY MONTHLY HYDRO CAP (TOTAL HYDRO LIMIT)
            # ---------------------------------------------
            hydro_cap = HYDRO_MONTHLY_CAP[month]

            hydro_mask = (available_type == 'HYDRO')
            thermal_mask = (available_type == 'THERMAL')

            hydro_indices = np.where(hydro_mask)[0]
            thermal_indices = np.where(thermal_mask)[0]

            # Separate hydro and thermal
            hydro_capacities = available_cap[hydro_indices]
            hydro_costs = available_cost[hydro_indices]

            thermal_capacities = available_cap[thermal_indices]
            thermal_costs = available_cost[thermal_indices]

            # -------------------------------------------------
            # 1. DISPATCH HYDRO FIRST (within hydro cap)
            # -------------------------------------------------
            hydro_dispatch = 0.0

            if len(hydro_capacities) > 0:

                hydro_order = np.argsort(hydro_costs)
                sorted_hydro_cap = hydro_capacities[hydro_order]

                remaining_hydro_cap = hydro_cap

                for cap in sorted_hydro_cap:
                    if remaining_hydro_cap <= 0:
                        break
                    supply = min(cap, remaining_hydro_cap)
                    hydro_dispatch += supply
                    remaining_hydro_cap -= supply

            # -------------------------------------------------
            # 2. DISPATCH THERMAL BY MERIT ORDER
            # -------------------------------------------------
            remaining_load = current_load - hydro_dispatch
            thermal_dispatch = 0.0

            if remaining_load > 0 and len(thermal_capacities) > 0:

                thermal_order = np.argsort(thermal_costs)
                sorted_thermal_cap = thermal_capacities[thermal_order]

                for cap in sorted_thermal_cap:
                    if remaining_load <= 0:
                        break
                    supply = min(cap, remaining_load)
                    thermal_dispatch += supply
                    remaining_load -= supply

            effective_generation = hydro_dispatch + thermal_dispatch

        # -------------------------------------------------
        # LOSS OF LOAD CHECK
        # -------------------------------------------------
        min_event_time = np.min(time_to_event)
        time_step = min(1.0, min_event_time)

        if effective_generation < current_load:
            total_LOL_hours += time_step

        # -------------------------------------------------
        # ADVANCE TIME
        # -------------------------------------------------
        current_time += time_step
        time_to_event -= time_step

        # -------------------------------------------------
        # HANDLE FAILURE / REPAIR EVENTS
        # -------------------------------------------------
        events = np.where(time_to_event <= 1e-6)[0]

        for i in events:
            state[i] = 1 - state[i]

            if state[i] == 0:
                # repaired → sample time to failure
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                # failed → sample repair time
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # -------------------------------------------------
        # PROGRESS PRINT
        # -------------------------------------------------
        if int(current_time) % (HOURS_PER_YEAR * 100) < time_step:
            print(f"Year {int(current_time // HOURS_PER_YEAR)} completed")

    # =========================================================
    # RESULTS
    # =========================================================

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

    Gen, MTTF, MTTR, GenType, UnitCost, Load = load_data()
    run_smcs(Gen, MTTF, MTTR, GenType, UnitCost, Load)

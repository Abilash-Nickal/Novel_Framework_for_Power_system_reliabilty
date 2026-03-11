import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================================================

NUM_YEARS = 1000  # Set to 30,000 for your final run
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
    [853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# 4. DATA LOADING & PRE-SORTING
# =========================================================

def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    # 1. Logic: Line up according to unit cost (Cheapest First)
    # Ensure your CSV has a column for cost (e.g., 'Variable Cost (LKR/kWh)')
    # If the column name is different, change it here.
    cost_col = 'Unit Cost (LKR/kWh)'
    if cost_col not in df_gen.columns:
        # Fallback for testing if column name is different
        cost_col = df_gen.columns[df_gen.columns.str.contains('Cost', case=False)][0]

    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True) # Sort by cost for merit order dispatch

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)
    GenType = df_gen['TYPES'].values.astype(str)
    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    print(f"Data Loaded. Units sorted by cost. Cheapest: {df_gen['Name'].iloc[0] if 'Name' in df_gen else 'Unit 0'}")
    return Gen, MTTF, MTTR, GenType, UnitCost, Annual_Load


# =========================================================
# 5. SEQUENTIAL MONTE CARLO SIMULATION
# =========================================================

def run_smcs(Gen, MTTF, MTTR, GenType, UnitCost, Load):
    n_gen = len(Gen)
    state = np.zeros(n_gen, dtype=int)  # 0 = UP, 1 = DOWN
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    total_system_cost = 0.0
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")
    start_clock = datetime.now()

    while current_time < TOTAL_HOURS:
        # Time indices
        hour_of_year = int(current_time) % HOURS_PER_YEAR
        month_idx = min(hour_of_year // 730, 11)  # Approximate month
        current_load = Load[hour_of_year]

        # ---------------------------------------------
        # 1. IDENTIFY AVAILABLE UNITS
        # ---------------------------------------------
        up_mask = (state == 0)

        # ---------------------------------------------
        # 2. MERIT-ORDER DISPATCH LOGIC
        # ---------------------------------------------
        # We iterate through the PRE-SORTED units to satisfy demand
        dispatched_power = 0.0
        hourly_cost = 0.0

        # Track hydro usage for the seasonal cap
        current_hydro_used = 0.0
        hydro_cap = HYDRO_MONTHLY_CAP[month_idx]

        for i in range(n_gen):
            if not up_mask[i]: continue  # Skip if unit is DOWN
            if dispatched_power >= current_load: break  # Stop if load is met

            needed = current_load - dispatched_power

            if GenType[i].upper() == 'HYDRO':
                # Apply seasonal cap logic
                can_provide_hydro = min(Gen[i], hydro_cap - current_hydro_used)
                actual_contribution = min(can_provide_hydro, needed)

                if actual_contribution > 0:
                    dispatched_power += actual_contribution
                    current_hydro_used += actual_contribution
                    hourly_cost += actual_contribution * UnitCost[i]
            else:
                # Thermal Dispatch
                actual_contribution = min(Gen[i], needed)
                dispatched_power += actual_contribution
                hourly_cost += actual_contribution * UnitCost[i]

        # ---------------------------------------------
        # 3. RELIABILITY & COST ACCUMULATION
        # ---------------------------------------------
        min_event_time = np.min(time_to_event)
        time_step = min(1.0, min_event_time)

        # Check for Loss of Load
        # Logic: If even after checking all available units we couldn't meet load
        if dispatched_power < current_load - 1e-6:  # Use tolerance for float
            total_LOL_hours += time_step

        total_system_cost += (hourly_cost * time_step)

        # ---------------------------------------------
        # 4. ADVANCE SYSTEM STATE
        # ---------------------------------------------
        current_time += time_step
        time_to_event -= time_step

        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            if state[i] == 0:  # Repaired
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:  # Failed
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # Progress log
        if int(current_time) % (HOURS_PER_YEAR * 100) < time_step:
            print(f"Progress: Year {int(current_time // HOURS_PER_YEAR)}...")

    # =================================================
    # RESULTS
    # =================================================
    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS
    avg_annual_cost = total_system_cost / NUM_YEARS
    runtime = datetime.now() - start_clock

    print("\n========= FINAL MERIT-ORDER RESULTS =========")
    print(f"Total Loss of Load Hours : {total_LOL_hours:.2f}")
    print(f"LOLE (Hours/Year)        : {LOLE:.4f}")
    print(f"LOLP                     : {LOLP:.8f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")
    print(f"Simulation Runtime       : {runtime}")


if __name__ == "__main__":
    Gen, MTTF, MTTR, GenType, UnitCost, Load = load_data()
    run_smcs(Gen, MTTF, MTTR, GenType, UnitCost, Load)
import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================================================
NUM_YEARS = 1000  # Change to 30,000 for final research results
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# =========================================================
# 2. INPUT FILES
# =========================================================
GEN_DATA_FILE = "../../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# =========================================================
# 3. MONTHLY HYDRO CAPACITY (MW)
# =========================================================
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# 4. DATA LOADING & PRE-PROCESSING
# =========================================================
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    # Merit-Order Logic: Sort by Unit Cost (Cheapest First)
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    # Convert to Numpy for speed
    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # Pre-convert type to boolean for performance in the inner loop
    is_hydro = df_gen['TYPES'].str.upper().str.strip() == 'HYDRO'
    is_hydro = is_hydro.values

    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # Precise Month Lookup (744h for Jan, 672h for Feb, etc.)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days):
        month_lookup.extend([m] * (days * 24))
    month_lookup = np.array(month_lookup)

    return Gen, MTTF, MTTR, is_hydro, UnitCost, Annual_Load, month_lookup


# =========================================================
# 5. OPTIMIZED SEQUENTIAL SIMULATION
# =========================================================
def run_smcs(Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup):
    n_gen = len(Gen)
    state = np.zeros(n_gen, dtype=int)  # 0=UP, 1=DOWN
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    total_system_cost = 0.0
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")
    start_clock = datetime.now()

    while current_time < TOTAL_HOURS:
        hour_of_year = int(current_time) % HOURS_PER_YEAR
        month_idx = month_lookup[hour_of_year]
        current_load = Load[hour_of_year]

        # ---------------------------------------------
        # MERIT-ORDER DISPATCH
        # ---------------------------------------------
        up_mask = (state == 0)
        dispatched_power = 0.0
        hourly_cost = 0.0
        current_hydro_used = 0.0
        h_cap = HYDRO_MONTHLY_CAP[month_idx]

        # Process units in pre-sorted merit order
        for i in range(n_gen):
            if not up_mask[i]: continue
            if dispatched_power >= current_load: break

            needed = current_load - dispatched_power

            if is_hydro[i]:
                # Combined logic: Plant health + Seasonal water limit
                potential = min(Gen[i], h_cap - current_hydro_used)
                contribution = min(potential, needed)
                if contribution > 0:
                    dispatched_power += contribution
                    current_hydro_used += contribution
                    hourly_cost += contribution * UnitCost[i]
            else:
                # Thermal Dispatch
                contribution = min(Gen[i], needed)
                dispatched_power += contribution
                hourly_cost += contribution * UnitCost[i]

        # ---------------------------------------------
        # METRICS & ADVANCE
        # ---------------------------------------------
        dt = min(1.0, np.min(time_to_event))

        if dispatched_power < current_load - 1e-4:
            total_LOL_hours += dt

        total_system_cost += (hourly_cost * dt)
        current_time += dt
        time_to_event -= dt

        # State Transitions
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            if state[i] == 0:  # To UP
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:  # To DOWN
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # Progress every 1000 years
        if int(current_time) % (HOURS_PER_YEAR * 1000) < dt:
            print(f"Progress: Year {int(current_time // HOURS_PER_YEAR)}...")

    # Final Calculation
    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS
    avg_annual_cost = total_system_cost / NUM_YEARS

    print("\n========= FINAL VALIDATED RESULTS =========")
    print(f"LOLE (Hours/Year)        : {LOLE:.4f}")
    print(f"LOLP                     : {LOLP:.8f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")
    print(f"Execution Time           : {datetime.now() - start_clock}")


if __name__ == "__main__":
    Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup = load_data()
    run_smcs(Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup)
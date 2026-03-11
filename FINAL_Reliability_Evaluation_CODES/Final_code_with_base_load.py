import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. SIMULATION CONFIGURATION ---
NUM_YEARS = 1000
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# 2. INPUT FILES
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# 3. MONTHLY HYDRO CAPACITY (MW)
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# 4. DATA LOADING
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    # ---- Merit Order Sorting (Cheapest First) ----
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # Faster flags
    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values  # Base Load = Coal
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # ---- Exact Month Lookup (Non-leap year) ----
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days):
        month_lookup.extend([m] * (days * 24))
    month_lookup = np.array(month_lookup)

    # Returning exactly 8 items
    return Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, month_lookup


# --- 5. SIMULATION ENGINE ---
def run_smcs(Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup):
    num_gen = len(Gen)

    # 0 = UP, 1 = DOWN
    state = np.zeros(num_gen, dtype=int)
    time_to_event = -MTTF * np.log(np.random.rand(num_gen))

    total_lol_hours = 0.0
    total_loee = 0.0
    total_system_cost = 0.0
    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...")
    start_time = datetime.now()

    while current_time < TOTAL_HOURS:
        h_year = int(current_time) % HOURS_PER_YEAR
        month_idx = month_lookup[h_year]
        current_load = Load[h_year]

        # --- DISPATCH LOGIC ---
        up_mask = (state == 0)
        dispatched = 0.0
        hourly_cost = 0.0
        hydro_used = 0.0

        # Now month_idx will correctly be an integer (0-11)
        h_cap = HYDRO_MONTHLY_CAP[month_idx]

        # STEP 1: BASE LOAD (COAL) ALWAYS FIRST
        for i in range(num_gen):
            if up_mask[i] and is_coal[i]:
                contribution = min(Gen[i], current_load - dispatched)
                dispatched += contribution
                hourly_cost += contribution * 1000 * UnitCost[i]
            if dispatched >= current_load: break

        # STEP 2: MERIT ORDER (HYDRO & THERMAL)
        for i in range(num_gen):
            if not up_mask[i] or is_coal[i]: continue
            if dispatched >= current_load: break

            needed = current_load - dispatched
            if is_hydro[i]:
                # Resource constraint (Water)
                potential = min(Gen[i], h_cap - hydro_used)
                contribution = min(potential, needed)
                if contribution > 0:
                    dispatched += contribution
                    hydro_used += contribution
                    hourly_cost += contribution * 1000 * UnitCost[i]
            else:
                contribution = min(Gen[i], needed)
                dispatched += contribution
                hourly_cost += contribution * 1000 * UnitCost[i]

        # --- SYSTEM ADVANCEMENT ---
        dt = min(1.0, np.min(time_to_event))

        if dispatched < current_load - 1e-4:
            total_lol_hours += dt
            total_loee += (current_load - dispatched) * dt

        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle Failures/Repairs
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]  # flip state

            # Assign next event time
            if state[i] == 0:
                # Just repaired, next event is FAILURE (using MTTF)
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                # Just failed, next event is REPAIR (using MTTR)
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

            # Progress every 1000 years
        if int(current_time) % (HOURS_PER_YEAR * 100) < dt:
            print(f"Progress: Year {int(current_time // HOURS_PER_YEAR)}, Time {current_time:.2f} hours, LOLP so far: {total_lol_hours / current_time:.8f}")
            print(f'"Current Load: {current_load:.2f} MW, Dispatched: {dispatched:.2f} MW, Hourly Cost: LKR {hourly_cost:,.2f}')
    # --- RESULTS ---
    LOLP = total_lol_hours / TOTAL_HOURS
    LOLE = total_lol_hours / NUM_YEARS
    LOEE = total_loee / NUM_YEARS
    avg_annual_cost = total_system_cost / NUM_YEARS


    print("\n========= FINAL RESULTS =========")
    print(f"LOLE (Hours/Year)        : {LOLE:.4f}")
    print(f"LOLP                     : {LOLP:.8f}")
    print(f"LOEE (MWh/Year)          : {LOEE:.2f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")

    # Fixed 'start_clock' to 'start_time'
    print(f"Execution Time           : {datetime.now() - start_time}")


if __name__ == "__main__":
    # FIXED: Unpacking variables exactly in the order returned by load_data()
    Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup = load_data()

    run_smcs(Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup)
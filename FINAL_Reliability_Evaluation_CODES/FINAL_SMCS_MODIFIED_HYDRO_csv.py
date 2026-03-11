import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
from datetime import datetime
import multiprocessing


# 1. RESEARCH CONFIGURATION

TOTAL_RESEARCH_YEARS = 1000
HOURS_PER_YEAR = 8760
NUM_CORES = 4  # Set to 4, 6, or 8 based on computer

# --- INPUT FILES ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED_2025.csv"
HYDRO_DATA_FILE = "../data/Monthly_Hydro_Profile.csv"



# 2. DATA PREPARATION and LOADING

def load_data():
    # 1. Load Generators
    genertors_data = pd.read_csv(GEN_DATA_FILE)
    cost_col = 'Unit Cost (LKR/kWh)'
    genertors_data = genertors_data.sort_values(by=cost_col).reset_index(drop=True)

    Gen = genertors_data['Unit Capacity (MW)'].values.astype(float)
    MTTF = genertors_data['MTTF (hours)'].values.astype(float)
    MTTR = genertors_data['MTTR (hours)'].values.astype(float)
    is_coal = (genertors_data['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (genertors_data['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = genertors_data[cost_col].values.astype(float)

    # 2. Load 8760 System Load Curve
    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float) # Assuming the first column contains the hourly load data
                                                        # other column contains the only solar,wind modified ,not modified hourly load data there

    # 3. Load 24x12 Hydro Profile CSV (Now supports Headings!)
    df_hydro = pd.read_csv(HYDRO_DATA_FILE)


    # It ensures we ONLY take the last 12 columns (Jan-Dec) for the math.
    if df_hydro.shape[1] > 12:
        df_hydro = df_hydro.iloc[:, -12:]

        # We use .T (Transpose) to flip the 24x12 CSV into a 12x24 matrix for the simulation
    Hydro_Matrix = df_hydro.values.T.astype(float)

    # Month mapping for 8760 hours
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, days in enumerate(month_days) for _ in range(days * 24)])

    # Pack everything into the tuple
    return (Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, month_lookup, Hydro_Matrix)


# 3. PARALLEL WORKER ENGINE

def worker_sim(worker_id, years_to_run, data_tuple):
    # Unpack the tuple
    Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup, Hydro_Matrix = data_tuple

    # Independent seeding for parallel universes
    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)

    num_gen = len(Gen)
    state = np.zeros(num_gen, dtype=int)

    # Safely initialize time to first event
    rand_vals = np.clip(np.random.rand(num_gen), 1e-9, 0.999999) # Avoid log(0) or log(1) which can cause issues in time_to_event calculation
    time_to_event = -MTTF * np.log(rand_vals)

    # Aggregators
    total_lol_h, total_loee, total_cost, total_events = 0.0, 0.0, 0.0, 0
    was_in_lol = False
    current_time = 0.0
    target_hours = years_to_run * HOURS_PER_YEAR

    while current_time < target_hours:
        # Time variables
        h_year = int(current_time) % HOURS_PER_YEAR # Hour of the year (0 to 8759)
        month_idx = month_lookup[h_year] # Month index (0 to 11) for this hour
        hour_of_day = int(current_time) % 24 # Hour of the day (0 to 23) for this hour

        # 1. GET TARGETS
        current_load = Load[h_year]
        target_hydro_mw = Hydro_Matrix[month_idx][hour_of_day]  # Uses the imported CSV data

        # Dispatch Variables
        up_gen = (state == 0) # Mask of generators that are currently UP
        dispatched = 0.0 # Total MW dispatched so far for this hour
        hourly_cost = 0.0 # Cost accumulated for this hour
        hydro_dispatched_total = 0.0 # Total Hydro dispatched so far for this hour (to enforce the hourly hydro limit from the CSV)

        # PASS 1: COAL (BASE LOAD)
        for i in range(num_gen):
            if up_gen[i] and is_coal[i]: # Coal plants are dispatched first as base load
                contrib = min(Gen[i], current_load - dispatched) # Contribution from this generator is the minimum of its capacity and the remaining load
                dispatched += contrib
                hourly_cost += contrib * 1000 * UnitCost[i]
            if dispatched >= current_load: break

        # PASS 2: HYDRO & THERMAL (MERIT ORDER)
        for i in range(num_gen): # Now we go through the generators in order of cost (since we sorted them at the beginning). We skip coal (already dispatched) and any that are down.
            if not up_gen[i] or is_coal[i]: continue # Skip generators that are down or already dispatched as coal
            if dispatched >= current_load: break

            needed = current_load - dispatched

            if is_hydro[i]: # This is a hydro generator, so we need to check the hourly hydro limit from the CSV
                # The remaining hydro allowance for this specific hour based on the CSV
                hydro_allowed_this_hour = max(0.0, target_hydro_mw - hydro_dispatched_total)

                contrib = min(Gen[i], hydro_allowed_this_hour, needed) # The contribution from this hydro generator is limited by its capacity, the remaining hydro allowance for this hour, and the remaining load needed
                if contrib > 0:
                    dispatched += contrib
                    hydro_dispatched_total += contrib
                    hourly_cost += contrib * 1000 * UnitCost[i]
            else:
                # Regular Thermal
                contrib = min(Gen[i], needed)
                dispatched += contrib
                hourly_cost += contrib * 1000 * UnitCost[i]

        # SYSTEM TIME ADVANCEMENT
        dt = max(min(1.0, np.min(time_to_event)), 1e-4)

        is_failing = dispatched < current_load - 1e-4
        if is_failing:
            total_lol_h += dt
            total_loee += (current_load - dispatched) * dt
            if not was_in_lol: total_events += 1

        was_in_lol = is_failing
        total_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle State Transitions (Failures & Repairs)
        expired = np.where(time_to_event <= 1e-6)[0] # Find all generators whose time to event has expired (either failure or repair)
        for i in expired:
            state[i] = 1 - state[i]
            ref = MTTF[i] if state[i] == 0 else MTTR[i]
            rand_val = max(1e-9, min(0.999999, np.random.rand()))
            time_to_event[i] = -ref * np.log(rand_val)

    return {"h": total_lol_h, "e": total_loee, "c": total_cost, "f": total_events}


# =========================================================
# 4. MASTER CONTROLLER
# =========================================================
if __name__ == '__main__':
    print(f"--- Starting Parallel SMCS (CSV Hydro Profiles) ---")
    data_tuple = load_data()
    start_wall = datetime.now()

    y_per_core = TOTAL_RESEARCH_YEARS // NUM_CORES

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(worker_sim, i, y_per_core, data_tuple) for i in range(NUM_CORES)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Final Aggregation
    agg_h = sum(r["h"] for r in results)
    agg_e = sum(r["e"] for r in results)
    agg_c = sum(r["c"] for r in results)
    agg_f = sum(r["f"] for r in results)

    # Calculate Indices
    LOLE = agg_h / TOTAL_RESEARCH_YEARS
    LOLP = agg_h / (TOTAL_RESEARCH_YEARS * HOURS_PER_YEAR)
    LOEE = agg_e / TOTAL_RESEARCH_YEARS
    LOLF = agg_f / TOTAL_RESEARCH_YEARS
    LOLD = agg_h / agg_f if agg_f > 0 else 0
    avg_annual_cost = agg_c / TOTAL_RESEARCH_YEARS

    # Print Results
    print("\n" + "=" * 40)
    print(f"RESULTS FOR {TOTAL_RESEARCH_YEARS} YEARS")
    print("=" * 40)
    print(f"LOLE : {LOLE:.4f} Hours/Year")
    print(f"LOLP : {LOLP:.8f}")
    print(f"LOEE : {LOEE:.2f} MWh/Year")
    print(f"LOLF : {LOLF:.4f} Occurrences/Year")
    print(f"LOLD : {LOLD:.4f} Hours/Occurrence")
    print(f"Cost : LKR {avg_annual_cost:,.2f} /Year")
    print(f"Total Sim Time: {datetime.now() - start_wall}")
    print("=" * 40)
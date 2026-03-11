from idlelib.config_key import AVAILABLE_KEYS

import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
from datetime import datetime
import multiprocessing

# --- 1. SIMULATION CONFIGURATION ---
TOTAL_RESEARCH_YEARS = 1000  # Total years you want to simulate
HOURS_PER_YEAR = 8760

# Set to an integer (e.g., 4, 6) to limit CPU usage, or leave as None to use all available cores
NUM_CORES_TO_USE = 4

# 2. INPUT FILES
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# --- 3.MONTHLY HYDRO CAPACITY (MW) FOR medium rainfall year (taken from CEB data) --- (there are 5 dry,wet,medium,v.dry, v.wet)
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# --- 4. DATA LOADING ---
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    cost_colm = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_colm).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_colm].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float) # Assuming the first column contains the hourly load data
                                                        # other column contains the only solar,wind modified ,not modified hourly load data there
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days):
        month_lookup.extend([m] * (days * 24))
    month_lookup = np.array(month_lookup)

    return Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, month_lookup


# --- 5. THE WORKER FUNCTION (Runs on a single CPU Core) ---
def worker_smcs(worker_id, years_to_run, data_tuple):

    # Unpack the data tuple (this is the same for all workers, so we pass it as a single argument to avoid redundant loading)
    Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup = data_tuple


    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)  # CRITICAL STEP: Re-seed the random number generator for this specific core!
                                                                                # If we don't do this, every core will simulate the exact same failures.

    num_of_gen = len(Gen) # Number of generators in the system
    state = np.zeros(num_of_gen, dtype=int)
    time_to_event = -MTTF * np.log(np.random.rand(num_of_gen)) # Time until next failure for each generator arry of time to event for each generator
    total_lol_hours = 0.0
    total_loee = 0.0
    total_system_cost = 0.0
    total_lol_events = 0
    was_in_lol = False

    current_time = 0.0
    target_sim_hrs = years_to_run * HOURS_PER_YEAR

    while current_time < target_sim_hrs: # Loop until we've simulated the required number of years for this worker
        hour_of_year = int(current_time) % HOURS_PER_YEAR # Hour of the year (0 to 8759)
        month_idx = month_lookup[hour_of_year]
        current_load = Load[hour_of_year]

        up_mask = (state == 0)
        AVAILABLE = 0.0
        hourly_cost = 0.0
        hydro_used = 0.0
        hydro_cap = HYDRO_MONTHLY_CAP[month_idx]

        # STEP 1: BASE LOAD (COAL)
        for i in range(num_of_gen):
            if up_mask[i] and is_coal[i]: # First, we check if the generator is up and if it's a coal generator (base load)
                contribution = min(Gen[i], current_load - AVAILABLE)
                AVAILABLE += contribution
                hourly_cost += contribution * 1000 * UnitCost[i] # Convert MW to kW for cost calculation
            if AVAILABLE >= current_load: break # If we've already met the load with coal, we can skip checking the rest of the generators for this hour

        # STEP 2: MERIT ORDER (HYDRO & THERMAL)
        for i in range(num_of_gen):
            if not up_mask[i] or is_coal[i]: continue # Skip generators that are down or already dispatched as base load
            if AVAILABLE >= current_load: break # No need to check further if we've already met the load

            needed = current_load - AVAILABLE

            # For hydro generators, we need to check the monthly capacity limit.
            if is_hydro[i]:
                from_hydro = min(Gen[i], hydro_cap - hydro_used) # Hydro can only contribute up to its monthly capacity limit, so we check how much of that capacity is still available after accounting for what we've already used this month
                contribution = min(from_hydro, needed)
                if contribution > 0:
                    AVAILABLE += contribution
                    hydro_used += contribution
                    hourly_cost += contribution * 1000 * UnitCost[i]

            # For non-coal, non-hydro generators (e.g., gas)
            else:
                contribution = min(Gen[i], needed) # For non-coal, non-hydro generators (e.g., gas), we just check how much they can contribute based on their capacity and the remaining load needed
                AVAILABLE += contribution
                hourly_cost += contribution * 1000 * UnitCost[i]

        # SYSTEM ADVANCEMENT
        dt = min(1.0, np.min(time_to_event)) # We advance the system by either 1 hour or until the next failure/repair event, whichever comes first
        is_failiure = AVAILABLE < current_load - 1e-4

        if is_failiure:
            total_lol_hours += dt
            total_loee += (current_load - AVAILABLE) * dt
            if not was_in_lol:
                total_lol_events += 1

        was_in_lol = is_failiure
        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle Failures/Repairs
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            if state[i] == 0:
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

    # Return the RAW sums for this worker's chunk of years
    return {
        "lol_hours": total_lol_hours,
        "loee": total_loee,
        "cost": total_system_cost,
        "events": total_lol_events
    }


# --- 6. MASTER CONTROLLER ---
if __name__ == '__main__':
    # NOTE: Multiprocessing requires the code to be inside an `if __name__ == '__main__':` block.

    print("\n" + "=" * 50)
    print(" PARALLEL SMCS ENGINE (SRI LANKA) ")
    print("=" * 50)

    data_tuple = load_data()

    # 1. Detect CPU Cores and apply user limit
    available_cores = multiprocessing.cpu_count()
    if NUM_CORES_TO_USE is not None:
        # Ensures you don't accidentally ask for more cores than you actually have
        num_cores = min(NUM_CORES_TO_USE, available_cores)
    else:
        num_cores = available_cores

    print(f"System detected {available_cores} logical CPU cores.")
    print(f"Simulation will utilize {num_cores} cores.")

    # 2. Divide the workload
    years_per_core = TOTAL_RESEARCH_YEARS // num_cores
    remainder_years = TOTAL_RESEARCH_YEARS % num_cores

    # Create a list of jobs. (e.g., if 30000 years and 4 cores, 3 cores get 7500, 1 core gets 7500 + remainder)
    jobs = []
    for i in range(num_cores):
        years_for_this_worker = years_per_core + (remainder_years if i == 0 else 0)
        jobs.append((i, years_for_this_worker, data_tuple))
        print(f" -> Worker {i + 1} assigned {years_for_this_worker:,} years.")

    print(f"\nStarting simulation of {TOTAL_RESEARCH_YEARS:,} total years...")
    start_time = datetime.now()

    # 3. Launch Parallel Universes
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all jobs to the executor
        futures = [executor.submit(worker_smcs, job[0], job[1], job[2]) for job in jobs]

        # As each core finishes its chunk, grab the results
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            print(f"  Worker {i + 1}/{num_cores} finished.")

    # 4. Aggregate the results from all cores
    print("\nAggregating multi-core results...")
    total_lol_hours = sum(res["lol_hours"] for res in results)
    total_loee = sum(res["loee"] for res in results)
    total_cost = sum(res["cost"] for res in results)
    total_events = sum(res["events"] for res in results)

    # 5. Calculate Final Indices
    LOLP = total_lol_hours / (TOTAL_RESEARCH_YEARS * HOURS_PER_YEAR)
    LOLE = total_lol_hours / TOTAL_RESEARCH_YEARS
    LOEE = total_loee / TOTAL_RESEARCH_YEARS
    LOLF = total_events / TOTAL_RESEARCH_YEARS
    LOLD = (total_lol_hours / total_events) if total_events > 0 else 0.0
    avg_annual_cost = total_cost / TOTAL_RESEARCH_YEARS

    # --- FINAL OUTPUT ---
    print("\n========= FINAL ACADEMIC RESULTS =========")
    print(f"Total Years Simulated    : {TOTAL_RESEARCH_YEARS:,}")
    print(f"LOLE (Hours/Year)        : {LOLE:.4f}")
    print(f"LOLP                     : {LOLP:.8f}")
    print(f"LOEE (MWh/Year)          : {LOEE:.2f}")
    print(f"LOLF (Events/Year)       : {LOLF:.4f}")
    print(f"LOLD (Hours/Event)       : {LOLD:.4f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")
    print(f"Execution Time           : {datetime.now() - start_time}")
    print("==========================================")
from ctypes.wintypes import HHOOK

import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
from datetime import datetime
import multiprocessing

# --- 1. SIMULATION CONFIGURATION ---
TOTAL_RESEARCH_YEARS = 30000  # Total years you want to simulate
HOURS_PER_YEAR = 8760

# --- MAXIMUM RAM OPTIMIZATION ---
# Keeps RAM usage near zero to prevent numpy._core._exceptions._ArrayMemoryError
BATCH_YEARS = 10

# Set to an integer (e.g., 4, 6) to limit CPU usage, or leave as None to use all available cores
NUM_CORES_TO_USE = 8

# --- 2. INPUT FILES ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED_2025.csv"

# NEW: The 24-Row, 12-Column CSV file for your Hourly Hydro limits
HYDRO_DATA_FILE = "../data/Monthly_Hydro_Profile.csv"


# --- 3. DATA LOADING & PRE-PROCESSING ---
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)

    cost_colm = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_colm).reset_index(drop=True)

    # Use float32 instead of default float64 to cut memory usage in half
    Gen = df_gen['Unit Capacity (MW)'].values.astype(np.float32)
    MTTF = df_gen['MTTF (hours)'].values.astype(np.float32)
    MTTR = df_gen['MTTR (hours)'].values.astype(np.float32)

    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_colm].values.astype(np.float32)

    # Load 8760 Load Curve
    LOAD_COLUMN = "Modified_LC"  # Modified_LC,old_modified,Total_LC
    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load[LOAD_COLUMN].values.astype(np.float32)


    # -------------------------------------------------------------
    # NEW: LOAD 24x12 HYDRO CSV AND EXPAND TO 8760-HOUR ARRAY
    # -------------------------------------------------------------
    try:
        df_hydro = pd.read_csv(HYDRO_DATA_FILE)
        # Smart Filter: If you added an "Hour" column on the left, ignore it
        if df_hydro.shape[1] > 12:
            df_hydro = df_hydro.iloc[:, -12:]
        # Extract the 24x12 grid as float32 to save RAM
        hydro_matrix = df_hydro.values.astype(np.float32)
    except FileNotFoundError:
        print(f"[Warning] '{HYDRO_DATA_FILE}' not found. Using default flat 400MW profile.")
        hydro_matrix = np.full((24, 12), 400.0, dtype=np.float32)

    # Map the 24x12 matrix to a flat 8,760 hour array
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Hydro_8760 = []

    for m, days in enumerate(month_days):
        # Grab the 24-hour profile for this specific month
        daily_profile = hydro_matrix[:, m]
        # Repeat that 24-hour profile for the number of days in the month
        month_profile = np.tile(daily_profile, days)
        Hydro_8760.extend(month_profile)

    Hydro_8760 = np.array(Hydro_8760, dtype=np.float32)

    # We now return Hydro_8760 instead of the old month_lookup
    return Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, Hydro_8760


# --- 4. THE WORKER FUNCTION (Vectorized & Runs on a single CPU Core) ---
def worker_smcs(worker_id, years_to_run, data_tuple):
    # Unpack the data tuple (Now receives the 8760 Hydro Profile)
    Gen_orig, MTTF_orig, MTTR_orig, is_hydro_orig, is_coal_orig, UnitCost_orig, Load, Hydro_8760 = data_tuple

    # CRITICAL STEP: Re-seed the random number generator for this specific core!
    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)

    # --- 4.1 PREPARE DISPATCH ORDER ---
    # Force Base Load (Coal) FIRST, then Merit Order for the rest.
    coal_idx = np.where(is_coal_orig)[0]
    non_coal_idx = np.where(~is_coal_orig)[0]
    dispatch_order = np.concatenate([coal_idx, non_coal_idx])

    # Reorder all matrices based on physical dispatch sequence
    Gen = Gen_orig[dispatch_order]
    MTTF = MTTF_orig[dispatch_order]
    MTTR = MTTR_orig[dispatch_order]
    is_hydro = is_hydro_orig[dispatch_order]
    UnitCost = UnitCost_orig[dispatch_order]
    num_gen = len(Gen)

    total_hours_to_run = years_to_run * HOURS_PER_YEAR

    # --- 4.2 GENERATE ALL FAILURE/REPAIR TRANSITIONS UP-FRONT ---
    transitions_list = []
    for i in range(num_gen):
        # Estimate number of alternating states needed for the worker's full timespan
        num_cycles = int(total_hours_to_run / (MTTF[i] + MTTR[i]) * 1.5) + 100

        ttf = -MTTF[i] * np.log(np.random.rand(num_cycles))
        ttr = -MTTR[i] * np.log(np.random.rand(num_cycles))

        durations = np.empty(num_cycles * 2, dtype=np.float32)
        durations[0::2] = ttf  # Even indices: Time to Failure
        durations[1::2] = ttr  # Odd indices: Time to Repair

        trans = np.cumsum(durations)

        # Safeguard: Ensure transitions cover the entire timespan for this worker
        while trans[-1] < total_hours_to_run:
            extra_ttf = -MTTF[i] * np.log(np.random.rand(10))
            extra_ttr = -MTTR[i] * np.log(np.random.rand(10))
            extra_durations = np.empty(20, dtype=np.float32)
            extra_durations[0::2] = extra_ttf
            extra_durations[1::2] = extra_ttr
            trans = np.concatenate([trans, trans[-1] + np.cumsum(extra_durations)])

        transitions_list.append(trans)

    # Trackers for this core
    total_lol_hours = 0.0
    total_loee = 0.0
    total_lol_events = 0
    total_system_cost = 0.0
    was_failing_prev_batch = False

    # --- 4.3 BATCH PROCESSING (Avoids RAM limits while keeping vectorization) ---
    num_batches = int(np.ceil(years_to_run / BATCH_YEARS))

    for b in range(num_batches):
        years_in_this_batch = min(BATCH_YEARS, years_to_run - b * BATCH_YEARS)
        hours_in_this_batch = years_in_this_batch * HOURS_PER_YEAR

        start_hour = b * BATCH_YEARS * HOURS_PER_YEAR
        end_hour = start_hour + hours_in_this_batch
        hours = np.arange(start_hour, end_hour)

        # Tile standard 8760 profiles into the batch timeframe
        Load_batch = np.tile(Load, years_in_this_batch)

        # NEW: Tile the exact Hourly Hydro limit into the batch timeframe
        H_cap_batch = np.tile(Hydro_8760, years_in_this_batch)

        # 1. Map continuous transition times to discrete hourly states (0 = DOWN, 1 = UP)
        States = np.zeros((num_gen, hours_in_this_batch), dtype=np.int8)
        for i in range(num_gen):
            indices = np.searchsorted(transitions_list[i], hours)
            States[i, :] = (indices % 2 == 0).astype(np.int8)

        # 2. Base hourly availability
        Avail = States * Gen[:, None]

        # 3. Vectorized Hydro Capping
        if np.any(is_hydro):
            hydro_avail = Avail[is_hydro, :]
            cum_hydro = np.cumsum(hydro_avail, axis=0)

            # Cap the cumulative sum to the specific hour's hydro budget
            cum_hydro_capped = np.minimum(cum_hydro, H_cap_batch)

            # Revert the cumulative sum to get individual unit available dispatch
            capped_hydro_avail = np.empty_like(hydro_avail)
            capped_hydro_avail[0, :] = cum_hydro_capped[0, :]
            if len(hydro_avail) > 1:
                capped_hydro_avail[1:, :] = cum_hydro_capped[1:, :] - cum_hydro_capped[:-1, :]

            # Overwrite standard Avail with strictly budget-capped Avail
            Avail[is_hydro, :] = capped_hydro_avail

        # 4. Vectorized Merit Order Dispatch
        cum_Avail = np.cumsum(Avail, axis=0)

        cum_Avail_prev = np.zeros_like(cum_Avail)
        cum_Avail_prev[1:, :] = cum_Avail[:-1, :]

        # Dispatched = min(Unit_Avail, Remaining_Load)
        Dispatched = np.minimum(Avail, np.maximum(0, Load_batch - cum_Avail_prev))

        # 5. Reliability Indices Evaluation
        Total_System_Avail = cum_Avail[-1, :]
        Shortfall = np.maximum(0, Load_batch - Total_System_Avail)
        is_failing = Shortfall > 1e-4

        total_lol_hours += np.sum(is_failing)
        total_loee += np.sum(Shortfall)

        # 6. Frequency Tracking (Find edge transitions True->False or False->True)
        is_failing_padded = np.insert(is_failing, 0, was_failing_prev_batch)
        new_events = is_failing & ~is_failing_padded[:-1]
        total_lol_events += np.sum(new_events)
        was_failing_prev_batch = is_failing[-1]

        # 7. Cost Evaluation
        hourly_costs = Dispatched * UnitCost[:, None] * 1000
        total_system_cost += np.sum(hourly_costs)

    # Return the RAW sums for this worker's chunk of years
    return {
        "lol_hours": total_lol_hours,
        "loee": total_loee,
        "cost": total_system_cost,
        "events": total_lol_events
    }


# --- 5. MASTER CONTROLLER ---
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print(" PARALLEL VECTORIZED SMCS ENGINE (SRI LANKA) ")
    print("=" * 50)

    data_tuple = load_data()

    # 1. Detect CPU Cores and apply user limit
    available_cores = multiprocessing.cpu_count()
    if NUM_CORES_TO_USE is not None:
        num_cores = min(NUM_CORES_TO_USE, available_cores)
    else:
        num_cores = available_cores

    print(f"System detected {available_cores} logical CPU cores.")
    print(f"Simulation will utilize {num_cores} cores.")

    # 2. Divide the workload
    years_per_core = TOTAL_RESEARCH_YEARS // num_cores
    remainder_years = TOTAL_RESEARCH_YEARS % num_cores

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
        futures = [executor.submit(worker_smcs, job[0], job[1], job[2]) for job in jobs]

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
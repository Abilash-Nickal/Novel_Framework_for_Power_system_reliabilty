import os
import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------
HOURS_PER_YEAR = 8760
# We process 10 years at a time. This keeps RAM usage low
# but runs thousands of times faster than an hourly loop.
BATCH_YEARS = 10


# ==========================================
# 1. DATA LOADING
# ==========================================
def load_system_data(gen_file, load_file):
    """
    Loads data for a BASIC SMCS (Pure Merit-Order).
    Matches the exact 2-argument signature your GUI expects.
    """
    df_gen = pd.read_csv(gen_file)
    cost_colm = 'Unit Cost (LKR/kWh)'

    # Basic Merit-Order Sort (Cheapest First)
    if cost_colm in df_gen.columns:
        df_gen = df_gen.sort_values(by=cost_colm).reset_index(drop=True)
        UnitCost = df_gen[cost_colm].values.astype(np.float32)
    else:
        UnitCost = np.zeros(len(df_gen), dtype=np.float32)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(np.float32)
    MTTF = df_gen['MTTF (hours)'].values.astype(np.float32)
    MTTR = df_gen['MTTR (hours)'].values.astype(np.float32)

    # Load 8760 Load Curve safely
    df_load = pd.read_csv(load_file)
    if "Modified_LC" in df_load.columns:
        Annual_Load = df_load["Modified_LC"].values.astype(np.float32)
    else:
        # Fallback to the last column if exact name isn't found
        Annual_Load = df_load.iloc[:, -1].values.astype(np.float32)

    # Return exactly what your GUI expects in "self.data"
    return {
        "Gen": Gen,
        "MTTF": MTTF,
        "MTTR": MTTR,
        "UnitCost": UnitCost,
        "Load": Annual_Load
    }


# ==========================================
# 2. VECTORIZED SIMULATION LOGIC
# ==========================================
def run_full_sequential_simulation(num_years, data, update_queue):
    """
    Fast Vectorized engine wrapper.
    Matches the exact 3-argument signature your GUI expects.
    """
    # Unpack pre-loaded data
    Gen = data["Gen"]
    MTTF = data["MTTF"]
    MTTR = data["MTTR"]
    UnitCost = data["UnitCost"]
    Load = data["Load"]
    num_gen = len(Gen)

    total_hours_to_run = num_years * HOURS_PER_YEAR
    start_wall_time = time.time()

    # Re-seed RNG for fresh runs
    np.random.seed(os.getpid() + int(time.time() * 1000) % 100000)

    # --- 1. GENERATE TRANSITIONS UP-FRONT ---
    transitions_list = []
    for i in range(num_gen):
        # Estimate number of cycles needed
        num_cycles = int(total_hours_to_run / (MTTF[i] + MTTR[i]) * 1.5) + 100
        ttf = -MTTF[i] * np.log(np.random.rand(num_cycles))
        ttr = -MTTR[i] * np.log(np.random.rand(num_cycles))

        durations = np.empty(num_cycles * 2, dtype=np.float32)
        durations[0::2] = ttf
        durations[1::2] = ttr

        trans = np.cumsum(durations)

        # Ensure we have enough states to cover the whole timeline
        while trans[-1] < total_hours_to_run:
            extra_ttf = -MTTF[i] * np.log(np.random.rand(10))
            extra_ttr = -MTTR[i] * np.log(np.random.rand(10))
            extra_durations = np.empty(20, dtype=np.float32)
            extra_durations[0::2] = extra_ttf
            extra_durations[1::2] = extra_ttr
            trans = np.concatenate([trans, trans[-1] + np.cumsum(extra_durations)])

        transitions_list.append(trans)

    # Tracking variables
    was_failing_prev_batch = False
    total_lol_hours = 0.0
    total_loee = 0.0
    total_cost = 0.0
    total_events = 0
    simulated_years = 0

    num_batches = int(np.ceil(num_years / BATCH_YEARS))

    # --- 2. BATCH PROCESSING LOOP ---
    for b in range(num_batches):
        years_in_this_batch = min(BATCH_YEARS, num_years - b * BATCH_YEARS)
        hours_in_this_batch = years_in_this_batch * HOURS_PER_YEAR

        start_hour = b * BATCH_YEARS * HOURS_PER_YEAR
        end_hour = start_hour + hours_in_this_batch
        hours = np.arange(start_hour, end_hour)

        Load_batch = np.tile(Load, years_in_this_batch)

        # Map continuous transitions to hourly discrete states
        States = np.zeros((num_gen, hours_in_this_batch), dtype=np.int8)
        for i in range(num_gen):
            indices = np.searchsorted(transitions_list[i], hours)
            States[i, :] = (indices % 2 == 0).astype(np.int8)

        # Base hourly availability (0 or MW Capacity)
        Avail = States * Gen[:, None]

        # Basic Merit Order Dispatch
        cum_Avail = np.cumsum(Avail, axis=0)
        cum_Avail_prev = np.zeros_like(cum_Avail)
        cum_Avail_prev[1:, :] = cum_Avail[:-1, :]

        Dispatched = np.minimum(Avail, np.maximum(0, Load_batch - cum_Avail_prev))

        # Reliability Calculation
        Total_System_Avail = cum_Avail[-1, :]
        Shortfall = np.maximum(0, Load_batch - Total_System_Avail)
        is_failing = Shortfall > 1e-4

        batch_lol_hours = np.sum(is_failing)
        batch_loee = np.sum(Shortfall)

        is_failing_padded = np.insert(is_failing, 0, was_failing_prev_batch)
        new_events = is_failing & ~is_failing_padded[:-1]
        batch_lol_events = np.sum(new_events)
        was_failing_prev_batch = is_failing[-1]

        hourly_costs = Dispatched * UnitCost[:, None] * 1000
        batch_cost = np.sum(hourly_costs)

        # Accumulate Totals
        total_lol_hours += float(batch_lol_hours)
        total_loee += float(batch_loee)
        total_events += int(batch_lol_events)
        total_cost += float(batch_cost)
        simulated_years += int(years_in_this_batch)

        # --- 3. SEND UPDATES TO GUI QUEUE ---
        if update_queue is not None:
            elapsed = time.time() - start_wall_time

            # Prevent division by zero
            lole = total_lol_hours / simulated_years if simulated_years > 0 else 0
            lolp = total_lol_hours / (simulated_years * HOURS_PER_YEAR) if simulated_years > 0 else 0
            loee = total_loee / simulated_years if simulated_years > 0 else 0
            avg_cost = total_cost / simulated_years if simulated_years > 0 else 0

            # Match exactly the dictionary keys your GUI loop expects
            update_queue.put({
                "y": simulated_years,
                "lole": lole,
                "lolp": lolp,
                "loee": loee,
                "events": total_events,
                "cost": avg_cost,
                "sim_time": elapsed,
                "done": False
            })

    # --- 4. FINISH SIMULATION ---
    if update_queue is not None:
        update_queue.put(
            {"done": True, "y": simulated_years, "lole": lole, "lolp": lolp, "loee": loee, "events": total_events,
             "cost": avg_cost, "sim_time": elapsed})

    return True
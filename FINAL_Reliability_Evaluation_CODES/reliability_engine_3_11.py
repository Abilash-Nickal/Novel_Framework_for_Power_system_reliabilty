import os
import time
import numpy as np
import pandas as pd


# ==========================================
# 1. WORKER FUNCTION (Vectorized + Live Updates + Stop Event)
# ==========================================
def worker_smcs(worker_id, years_to_run, batch_years, data_tuple, result_queue, stop_event):
    """
    The vectorized worker function. Runs on a single CPU core.
    Sends batch updates back to the GUI via result_queue.
    Listens to stop_event to gracefully exit if the user cancels.
    """
    if years_to_run <= 0:
        if result_queue is not None:
            result_queue.put({"type": "done", "worker_id": worker_id})
        return True

    HOURS_PER_YEAR = 8760

    # Unpack the data tuple
    Gen_orig, MTTF_orig, MTTR_orig, is_hydro_orig, is_coal_orig, UnitCost_orig, Load, Hydro_8760 = data_tuple

    # Re-seed the random number generator for this specific core
    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)

    # --- PREPARE DISPATCH ORDER ---
    coal_idx = np.where(is_coal_orig)[0]
    non_coal_idx = np.where(~is_coal_orig)[0]
    dispatch_order = np.concatenate([coal_idx, non_coal_idx])

    Gen = Gen_orig[dispatch_order]
    MTTF = MTTF_orig[dispatch_order]
    MTTR = MTTR_orig[dispatch_order]
    is_hydro = is_hydro_orig[dispatch_order]
    UnitCost = UnitCost_orig[dispatch_order]
    num_gen = len(Gen)

    total_hours_to_run = years_to_run * HOURS_PER_YEAR

    # --- GENERATE TRANSITIONS UP-FRONT ---
    transitions_list = []
    for i in range(num_gen):
        # CHECK STOP EVENT during heavy pre-calculation
        if stop_event is not None and stop_event.is_set():
            if result_queue is not None:
                result_queue.put({"type": "done", "worker_id": worker_id})
            return False

        num_cycles = int(total_hours_to_run / (MTTF[i] + MTTR[i]) * 1.5) + 100
        ttf = -MTTF[i] * np.log(np.random.rand(num_cycles))
        ttr = -MTTR[i] * np.log(np.random.rand(num_cycles))

        durations = np.empty(num_cycles * 2, dtype=np.float32)
        durations[0::2] = ttf
        durations[1::2] = ttr

        trans = np.cumsum(durations)

        while trans[-1] < total_hours_to_run:
            extra_ttf = -MTTF[i] * np.log(np.random.rand(10))
            extra_ttr = -MTTR[i] * np.log(np.random.rand(10))
            extra_durations = np.empty(20, dtype=np.float32)
            extra_durations[0::2] = extra_ttf
            extra_durations[1::2] = extra_ttr
            trans = np.concatenate([trans, trans[-1] + np.cumsum(extra_durations)])

        transitions_list.append(trans)

    was_failing_prev_batch = False

    # --- BATCH PROCESSING ---
    num_batches = int(np.ceil(years_to_run / batch_years))

    for b in range(num_batches):
        # CHECK STOP EVENT between batches
        if stop_event is not None and stop_event.is_set():
            break

        years_in_this_batch = min(batch_years, years_to_run - b * batch_years)
        hours_in_this_batch = years_in_this_batch * HOURS_PER_YEAR

        start_hour = b * batch_years * HOURS_PER_YEAR
        end_hour = start_hour + hours_in_this_batch
        hours = np.arange(start_hour, end_hour)

        Load_batch = np.tile(Load, years_in_this_batch)
        H_cap_batch = np.tile(Hydro_8760, years_in_this_batch)

        States = np.zeros((num_gen, hours_in_this_batch), dtype=np.int8)
        for i in range(num_gen):
            indices = np.searchsorted(transitions_list[i], hours)
            States[i, :] = (indices % 2 == 0).astype(np.int8)

        Avail = States * Gen[:, None]

        if np.any(is_hydro):
            hydro_avail = Avail[is_hydro, :]
            cum_hydro = np.cumsum(hydro_avail, axis=0)
            cum_hydro_capped = np.minimum(cum_hydro, H_cap_batch)

            capped_hydro_avail = np.empty_like(hydro_avail)
            capped_hydro_avail[0, :] = cum_hydro_capped[0, :]
            if len(hydro_avail) > 1:
                capped_hydro_avail[1:, :] = cum_hydro_capped[1:, :] - cum_hydro_capped[:-1, :]

            Avail[is_hydro, :] = capped_hydro_avail

        cum_Avail = np.cumsum(Avail, axis=0)
        cum_Avail_prev = np.zeros_like(cum_Avail)
        cum_Avail_prev[1:, :] = cum_Avail[:-1, :]

        Dispatched = np.minimum(Avail, np.maximum(0, Load_batch - cum_Avail_prev))

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

        # SEND DELTA TO GUI
        if result_queue is not None:
            result_queue.put({
                "type": "update",
                "worker_id": worker_id,
                "years": int(years_in_this_batch),
                "lol_hours": float(batch_lol_hours),
                "loee": float(batch_loee),
                "events": int(batch_lol_events),
                "cost": float(batch_cost)
            })

    # Signal that this worker has cleanly finished (or was cleanly stopped)
    if result_queue is not None:
        result_queue.put({"type": "done", "worker_id": worker_id})
    return True


# ==========================================
# 2. DATA LOADER
# ==========================================
def load_data(gen_file, load_file, hydro_file):
    df_gen = pd.read_csv(gen_file)
    cost_colm = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_colm).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(np.float32)
    MTTF = df_gen['MTTF (hours)'].values.astype(np.float32)
    MTTR = df_gen['MTTR (hours)'].values.astype(np.float32)

    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_colm].values.astype(np.float32)

    LOAD_COLUMN = "Modified_LC"
    df_load = pd.read_csv(load_file)
    Annual_Load = df_load[LOAD_COLUMN].values.astype(np.float32)

    try:
        df_hydro = pd.read_csv(hydro_file)
        if df_hydro.shape[1] > 12:
            df_hydro = df_hydro.iloc[:, -12:]
        hydro_matrix = df_hydro.values.astype(np.float32)
    except FileNotFoundError:
        print(f"[Warning] '{hydro_file}' not found. Using default flat 400MW profile.")
        hydro_matrix = np.full((24, 12), 400.0, dtype=np.float32)

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Hydro_8760 = []
    for m, days in enumerate(month_days):
        daily_profile = hydro_matrix[:, m]
        month_profile = np.tile(daily_profile, days)
        Hydro_8760.extend(month_profile)

    Hydro_8760 = np.array(Hydro_8760, dtype=np.float32)
    return Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, Hydro_8760
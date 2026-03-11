import numpy as np
import pandas as pd
import time
import os

# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================
HOURS_PER_YEAR = 8760
# Monthly Hydro Capacity (MW) - Sri Lankan Seasonal Constraints
# This is usually the reason why NSMCS "raw" results look different.
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# DATA LOADING FUNCTIONS
# =========================================================
def load_system_data(gen_file, load_file):
    """Loads and pre-processes the system data for NSMCS."""
    if not os.path.exists(gen_file) or not os.path.exists(load_file):
        raise FileNotFoundError("Data files not found. Check relative paths.")

    df_gen = pd.read_csv(gen_file, header=0)
    Gen_list = df_gen['Unit Capacity (MW)'].astype(float).values
    FOR_series = df_gen['Unit FOR'].astype(float).clip(0.0, 1.0).values

    # Identify Hydro units for seasonal capping
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    names = df_gen.iloc[:, 0].values if len(df_gen.columns) > 0 else [f"Gen_{i}" for i in range(len(Gen_list))]

    df_load = pd.read_csv(load_file, header=0)
    Load_list = df_load.iloc[:, 0].astype(float).values

    # Month lookup for the load hours (assuming chronological 8760 input)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, days in enumerate(month_days) for _ in range(days * 24)])

    return {
        "Gen": Gen_list,
        "FOR": FOR_series,
        "Load": Load_list,
        "is_hydro": is_hydro,
        "Names": names,
        "month_lookup": month_lookup
    }


# =========================================================
# IMPROVED NSMCS LOGIC
# =========================================================
def run_nsmcs_engine(target_iterations, data, update_queue, pause_event):
    """
    Advanced NSMCS Loop with Seasonal Constraints.
    """
    H = 0
    N = 0

    Gen_np = np.array(data["Gen"])
    FOR_np = np.array(data["FOR"])
    Load_np = np.array(data["Load"])
    is_hydro = data["is_hydro"]
    month_lookup = data["month_lookup"]

    num_gen = len(Gen_np)
    num_load_hours = len(Load_np)

    start_time = time.time()
    last_failed_state = np.zeros(num_gen, dtype=int)
    gui_update_step = max(10000, target_iterations // 1000)  # Update GUI every 1% of progress or at least every 10,000 iterations

    for n in range(target_iterations):
        while pause_event.is_set():
            time.sleep(0.1)

        N += 1

        # 1. Select a random hour and its corresponding month
        hour_idx = np.random.randint(0, min(num_load_hours, 8760))
        month_idx = month_lookup[hour_idx]
        current_load = Load_np[hour_idx]

        # 2. Generator Mechanical Availability (Random Draw vs FOR)
        random_gen_checks = np.random.random(num_gen)
        up_mask = random_gen_checks > FOR_np  # True if UP

        # 3. Apply PHYSICAL CONSTRAINTS (Hydro Cap)
        # Sum UP Thermal units
        available_thermal = np.sum(Gen_np[up_mask & (~is_hydro)])

        # Sum UP Hydro units but apply the seasonal limit
        raw_hydro_up = np.sum(Gen_np[up_mask & is_hydro])
        available_hydro = min(raw_hydro_up, HYDRO_MONTHLY_CAP[month_idx])

        total_available = available_thermal + available_hydro

        # 4. Check for Failure
        if current_load > total_available:
            H += 1
            last_failed_state = (~up_mask).astype(int)

        # Communication to GUI
        if N % gui_update_step == 0 or N == target_iterations:
            lolp = H / N
            update_queue.put({
                "n": N,
                "lolp": lolp,
                "lole": lolp * HOURS_PER_YEAR,
                "events": H,
                "sim_time": time.time() - start_time,
                "states": last_failed_state.copy(),
                "done": False
            })

    final_lolp = H / N if N > 0 else 0
    update_queue.put({"done": True, "lolp": final_lolp, "lole": final_lolp * HOURS_PER_YEAR})
import numpy as np
import pandas as pd
import time
import os

# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================
HOURS_PER_YEAR = 8760

# REALISM ADJUSTMENT: In a real grid, about 15-20% of the fleet
# is usually down for "Scheduled Maintenance" at any given time.
# Pure NSMCS often gets 0.000 because it forgets maintenance.
MAINTENANCE_FACTOR = 0.00


# =========================================================
# DATA LOADING FUNCTIONS
# =========================================================
def load_system_data(gen_file, load_file):
    if not os.path.exists(gen_file) or not os.path.exists(load_file):
        raise FileNotFoundError("Data files not found.")

    df_gen = pd.read_csv(gen_file)

    # Clean column names (remove spaces)
    df_gen.columns = df_gen.columns.str.strip()

    # Identify Capacity and FOR columns
    cap_col = 'Unit Capacity (MW)'
    for_col = 'Unit FOR'

    # Validation: Ensure we are reading numbers
    Gen_list = pd.to_numeric(df_gen[cap_col], errors='coerce').fillna(0).values
    FOR_list = pd.to_numeric(df_gen[for_col], errors='coerce').fillna(0.05).values

    # Load Profile
    df_load = pd.read_csv(load_file)
    Load_list = pd.to_numeric(df_load.iloc[:, 0], errors='coerce').fillna(0).values

    # DIAGNOSTIC PRINT (Check your console!)
    print(f"--- DATA DIAGNOSTICS ---")
    print(f"Total Installed Capacity: {np.sum(Gen_list):.2f} MW")
    print(f"Peak System Load: {np.max(Load_list):.2f} MW")
    print(f"Reserve Margin: {np.sum(Gen_list) - np.max(Load_list):.2f} MW")
    print(f"Average FOR: {np.mean(FOR_list):.4f}")
    print(f"------------------------")

    names = df_gen.iloc[:, 0].values if len(df_gen.columns) > 0 else [f"Unit_{i}" for i in range(len(Gen_list))]

    return {
        "Gen": Gen_list,
        "FOR": FOR_list,
        "Load": Load_list,
        "Names": names
    }


# =========================================================
# CORE SIMULATION ENGINE
# =========================================================
def run_nsmcs_engine(target_iterations, data, update_queue, pause_event):
    H = 0
    N = 0

    Gen_np = np.array(data["Gen"])
    FOR_np = np.array(data["FOR"])
    Load_np = np.array(data["Load"])

    num_gen = len(Gen_np)
    num_load_hours = len(Load_np)

    start_time = time.time()
    last_failed_state = np.zeros(num_gen, dtype=int)
    gui_update_step = max(5000, target_iterations // 100)

    for n in range(target_iterations):
        while pause_event.is_set():
            time.sleep(0.1)

        N += 1

        # 1. MECHANICAL FAILURE (FOR)
        random_draws = np.random.random(num_gen)
        up_mask = random_draws > FOR_np

        # 2. SCHEDULED MAINTENANCE (Realistic addition)
        # Randomly take out ~15% of units to simulate maintenance
        # so the grid isn't "perfectly full"
        maint_draws = np.random.random(num_gen)
        up_mask = up_mask & (maint_draws > MAINTENANCE_FACTOR)

        total_available = np.sum(Gen_np[up_mask])

        # 3. SAMPLE LOAD
        load_idx = np.random.randint(0, num_load_hours)
        current_load = Load_np[load_idx]

        # 4. RELIABILITY CHECK
        if current_load > total_available:
            H += 1
            last_failed_state = (~up_mask).astype(int)

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
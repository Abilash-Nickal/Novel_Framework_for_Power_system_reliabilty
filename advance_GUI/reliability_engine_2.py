import numpy as np
import pandas as pd
import time

# =========================================================
# 1. CONSTANTS & CONFIGURATION
# =========================================================
HOURS_PER_YEAR = 8760
# Monthly Hydro Capacity (MW) - Sri Lankan Seasonal Constraints
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# 2. DATA LOADING
# =========================================================
def load_system_data(gen_file, load_file):
    """Loads and pre-processes the system data with Base Load identification."""
    df_gen = pd.read_csv(gen_file)
    cost_col = 'Unit Cost (LKR/kWh)'

    # Merit-Order Sort (Cheapest First) for general dispatch
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # Identify Types
    # Base Load = Coal
    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values

    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(load_file)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # Exact Month Lookup Array
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, days in enumerate(month_days) for _ in range(days * 24)])

    return {
        "Gen": Gen, "MTTF": MTTF, "MTTR": MTTR,
        "is_hydro": is_hydro, "is_coal": is_coal,
        "UnitCost": UnitCost, "Load": Annual_Load,
        "month_lookup": month_lookup
    }


# =========================================================
# 3. COMPLETE SIMULATION LOGIC (The "Brain")
# =========================================================
def run_full_sequential_simulation(num_years, data, update_queue=None):
    """
    SMCS logic loop with Base Load (Coal) Priority Dispatch.
    """
    num_gen = len(data["Gen"])
    state = np.zeros(num_gen, dtype=int)  # 0 = UP, 1 = DOWN
    time_to_event = -data["MTTF"] * np.log(np.random.rand(num_gen))

    total_lol_hours = 0.0
    total_loee = 0.0
    total_system_cost = 0.0
    total_lol_events = 0
    was_in_lol = False

    current_time = 0.0
    total_target_hours = num_years * HOURS_PER_YEAR
    last_reported_year = 0
    start_wall_time = time.time()

    while current_time < total_target_hours:
        h_year = int(current_time) % HOURS_PER_YEAR
        month_idx = data["month_lookup"][h_year] #
        current_load = data["Load"][h_year]

        up_mask = (state == 0) #
        dispatched_power = 0.0
        hourly_cost = 0.0
        current_hydro_used = 0.0
        h_cap = HYDRO_MONTHLY_CAP[month_idx]

        # --- STEP 1: BASE LOAD DISPATCH (COAL ALWAYS FIRST) ---
        for i in range(num_gen):
            if not up_mask[i] or not data["is_coal"][i]:
                continue

            if dispatched_power >= current_load:
                break

            needed = current_load - dispatched_power
            contribution = min(data["Gen"][i], needed)

            dispatched_power += contribution
            hourly_cost += contribution * 1000 * data["UnitCost"][i]

        # --- STEP 2: MERIT ORDER DISPATCH (OTHERS) ---
        for i in range(num_gen):
            # Skip Coal (already handled) or DOWN units
            if not up_mask[i] or data["is_coal"][i]:
                continue

            if dispatched_power >= current_load:
                break

            needed = current_load - dispatched_power

            if data["is_hydro"][i]:
                # Apply Hydro Seasonality Cap
                potential = min(data["Gen"][i], h_cap - current_hydro_used)
                contribution = min(potential, needed)
                if contribution > 0:
                    dispatched_power += contribution
                    current_hydro_used += contribution
                    hourly_cost += contribution * 1000 * data["UnitCost"][i]
            else:
                # Other Thermal (Oil/Diesel) in merit order
                contribution = min(data["Gen"][i], needed)
                dispatched_power += contribution
                hourly_cost += contribution * 1000 * data["UnitCost"][i]

        # --- EVENT-BASED TIME STEP ---
        min_event = np.min(time_to_event)
        dt = min(1.0, min_event)

        # --- RELIABILITY CALCULATION ---
        is_failing = dispatched_power < current_load - 1e-4
        if is_failing:
            total_lol_hours += dt
            total_loee += (current_load - dispatched_power) * dt
            if not was_in_lol:
                total_lol_events += 1

        was_in_lol = is_failing
        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # --- STATE TRANSITIONS ---
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            ref = data["MTTF"][i] if state[i] == 0 else data["MTTR"][i]
            time_to_event[i] = -ref * np.log(np.random.rand())

        # --- GUI UPDATE ---
        cur_year = int(current_time // HOURS_PER_YEAR)
        if update_queue and cur_year > last_reported_year and cur_year % 10 == 0:
            elapsed = time.time() - start_wall_time
            update_queue.put({
                "y": cur_year,
                "lole": total_lol_hours / cur_year,
                "lolp": total_lol_hours / (cur_year * HOURS_PER_YEAR),
                "loee": total_loee / cur_year,
                "cost": total_system_cost / cur_year,
                "events": total_lol_events,
                "sim_time": elapsed,
                "done": False
            })
            last_reported_year = cur_year

    # Final Result
    final_results = {
        "lole": total_lol_hours / num_years,
        "lolp": total_lol_hours / total_target_hours,
        "loee": total_loee / num_years,
        "cost": total_system_cost / num_years,
        "events": total_lol_events,
        "sim_time": time.time() - start_wall_time,
        "done": True
    }
    if update_queue:
        update_queue.put(final_results)
    return final_results
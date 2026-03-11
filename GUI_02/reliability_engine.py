import numpy as np
import pandas as pd
import time

# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================
HOURS_PER_YEAR = 8760
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])

# SPEED OPTIMIZATION: How often to update the GUI (In Years)
# Increase this number (e.g., to 100) to make the simulation run much faster
GUI_UPDATE_STEP = 50

def load_system_data(gen_file, load_file):
    df_gen = pd.read_csv(gen_file)
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)
    return {
        "Gen": df_gen['Unit Capacity (MW)'].values.astype(float),
        "MTTF": df_gen['MTTF (hours)'].values.astype(float),
        "MTTR": df_gen['MTTR (hours)'].values.astype(float),
        "is_hydro": (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values,
        "is_coal": (df_gen['TYPES'].str.upper().str.contains('COAL')).values,
        "UnitCost": df_gen[cost_col].values.astype(float),
        "Load": pd.read_csv(load_file).iloc[:, 0].astype(float).values[:HOURS_PER_YEAR],
        "month_lookup": np.array(
            [m for m, days in enumerate([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) for _ in range(days * 24)])
    }


def run_full_sequential_simulation(num_years, data, update_queue=None):
    num_gen = len(data["Gen"])
    state = np.zeros(num_gen, dtype=int)
    time_to_event = -data["MTTF"] * np.log(np.random.rand(num_gen))

    total_lol_hours, total_loee, total_system_cost, total_lol_events = 0.0, 0.0, 0.0, 0
    was_in_lol = False
    current_time = 0.0
    total_target_hours = num_years * HOURS_PER_YEAR
    last_reported_year = 0
    start_wall_time = time.time()

    while current_time < total_target_hours:
        h_year = int(current_time) % HOURS_PER_YEAR
        month_idx = data["month_lookup"][h_year]
        current_load = data["Load"][h_year]

        # Dispatch
        up_mask = (state == 0)
        dispatched, hourly_cost, hydro_used = 0.0, 0.0, 0.0
        h_cap = HYDRO_MONTHLY_CAP[month_idx]

        # Pass 1: Coal
        for i in range(num_gen):
            if up_mask[i] and data["is_coal"][i]:
                contribution = min(data["Gen"][i], current_load - dispatched)
                dispatched += contribution
                hourly_cost += contribution * 1000 * data["UnitCost"][i]
            if dispatched >= current_load: break

        # Pass 2: Others
        for i in range(num_gen):
            if not up_mask[i] or data["is_coal"][i]: continue
            if dispatched >= current_load: break
            needed = current_load - dispatched
            if data["is_hydro"][i]:
                potential = min(data["Gen"][i], h_cap - hydro_used)
                contribution = min(potential, needed)
                if contribution > 0:
                    dispatched += contribution
                    hydro_used += contribution
                    hourly_cost += contribution * 1000 * data["UnitCost"][i]
            else:
                contribution = min(data["Gen"][i], needed)
                dispatched += contribution
                hourly_cost += contribution * 1000 * data["UnitCost"][i]

        dt = min(1.0, np.min(time_to_event))
        is_failing = dispatched < current_load - 1e-4
        if is_failing:
            total_lol_hours += dt
            total_loee += (current_load - dispatched) * dt
            if not was_in_lol: total_lol_events += 1

        was_in_lol = is_failing
        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            ref = data["MTTF"][i] if state[i] == 0 else data["MTTR"][i]
            time_to_event[i] = -ref * np.log(np.random.rand())

        # OPTIMIZED UPDATE CHECK
        cur_year = int(current_time // HOURS_PER_YEAR)
        if update_queue and cur_year >= last_reported_year + GUI_UPDATE_STEP:
            update_queue.put({
                "y": cur_year, "lole": total_lol_hours / cur_year,
                "lolp": total_lol_hours / (cur_year * HOURS_PER_YEAR),
                "loee": total_loee / cur_year, "cost": total_system_cost / cur_year,
                "events": total_lol_events, "sim_time": time.time() - start_wall_time,
                "states": state.copy(), "done": False
            })
            last_reported_year = cur_year

    final_results = {
        "lole": total_lol_hours / num_years, "lolp": total_lol_hours / total_target_hours,
        "loee": total_loee / num_years, "cost": total_system_cost / num_years,
        "events": total_lol_events, "sim_time": time.time() - start_wall_time, "done": True
    }
    if update_queue: update_queue.put(final_results)
    return final_results
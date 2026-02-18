import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# 1. CORE CONFIGURATION
# =========================================================
HOURS_PER_YEAR = 8760
# Monthly Hydro Capacity (MW) - Seasonal Constraints
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# 2. DATA LOADING LOGIC
# =========================================================
def load_system_data(gen_file, load_file):
    """Loads and pre-processes the Sri Lankan Grid data."""
    df_gen = pd.read_csv(gen_file)
    cost_col = 'Unit Cost (LKR/kWh)'

    # Sort by Merit Order (Cheapest First)
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(load_file)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # Exact Month Lookup (Standard 8760-hour year)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, days in enumerate(month_days) for _ in range(days * 24)])

    return {
        "Gen": Gen, "MTTF": MTTF, "MTTR": MTTR,
        "is_hydro": is_hydro, "UnitCost": UnitCost,
        "Load": Annual_Load, "month_lookup": month_lookup
    }


# =========================================================
# 3. CORE DISPATCH LOGIC (The "Brain")
# =========================================================
def run_dispatch_step(current_state, hour_load, month_idx, data):
    """Performs Merit-Order Dispatch for a single hour."""
    num_gen = len(data["Gen"])
    up_mask = (current_state == 0)

    dispatched_power = 0.0
    hourly_cost = 0.0
    current_hydro_used = 0.0
    h_cap = HYDRO_MONTHLY_CAP[month_idx]

    for i in range(num_gen):
        if not up_mask[i]: continue
        if dispatched_power >= hour_load: break

        needed = hour_load - dispatched_power

        if data["is_hydro"][i]:
            potential = min(data["Gen"][i], h_cap - current_hydro_used)
            contribution = min(potential, needed)
            if contribution > 0:
                dispatched_power += contribution
                current_hydro_used += contribution
                hourly_cost += contribution * 1000 * data["UnitCost"][i]
        else:
            contribution = min(data["Gen"][i], needed)
            dispatched_power += contribution
            hourly_cost += contribution * 1000 * data["UnitCost"][i]

    return dispatched_power, hourly_cost
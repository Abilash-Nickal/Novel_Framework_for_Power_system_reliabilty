import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. SIMULATION CONFIGURATION ---
NUM_YEARS = 100
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# Monthly Hydro Capacity Cap (MW) - Simplified 4-season representation
# Jan-Mar: 400 | Apr-Jun: 600 | Jul-Sep: 500 | Oct-Dec: 800
HYDRO_MONTHLY_CAP = [400, 400, 400, 600, 600, 600, 500, 500, 500, 800, 800, 800]


# --- 2. DUMMY DATA INITIALIZATION ---
def get_dummy_data():
    # 10 Generators: 1 Coal (Base), 3 Hydro (Limited), 6 Thermal (Peakers)
    data = {
        'Name': ['Coal_1', 'Hydro_1', 'Hydro_2', 'Hydro_3', 'Oil_1', 'Oil_2', 'Oil_3', 'Gas_1', 'Gas_2', 'Diesel_1'],
        'Capacity': [300, 200, 200, 150, 100, 100, 100, 150, 150, 50],
        'MTTF': [1000, 2500, 2500, 2500, 800, 800, 800, 1200, 1200, 500],
        'MTTR': [100, 50, 50, 50, 40, 40, 40, 60, 60, 24],
        'Cost': [18.5, 2.5, 2.7, 3.0, 65.0, 68.0, 72.0, 45.0, 48.0, 95.0],
        'Type': ['COAL', 'HYDRO', 'HYDRO', 'HYDRO', 'OIL', 'OIL', 'OIL', 'GAS', 'GAS', 'DIESEL']
    }
    df_gen = pd.DataFrame(data)

    # Sort by Merit Order (Cheapest First)
    df_gen = df_gen.sort_values(by='Cost').reset_index(drop=True)

    Gen = df_gen['Capacity'].values.astype(float)
    MTTF = df_gen['MTTF'].values.astype(float)
    MTTR = df_gen['MTTR'].values.astype(float)
    UnitCost = df_gen['Cost'].values.astype(float)
    is_hydro = (df_gen['Type'] == 'HYDRO').values
    is_coal = (df_gen['Type'] == 'COAL').values

    # Simple Load Profile (Repeating 24-hour cycle)
    # Peak at 19:00 (7 PM)
    daily_load = [600, 550, 500, 480, 500, 650, 800, 950, 1000, 1050, 1100, 1150,
                  1100, 1050, 1000, 1100, 1200, 1400, 1550, 1600, 1500, 1300, 1000, 800]
    Annual_Load = np.tile(daily_load, 365)[:HOURS_PER_YEAR]

    # Month lookup
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, d in enumerate(month_days) for _ in range(d * 24)])

    return Gen, MTTF, MTTR, UnitCost, is_hydro, is_coal, Annual_Load, month_lookup


# --- 3. SIMULATION ENGINE ---
def run_dummy_smcs():
    Gen, MTTF, MTTR, UnitCost, is_hydro, is_coal, Load, month_lookup = get_dummy_data()
    num_gen = len(Gen)

    # 0 = UP, 1 = DOWN
    state = np.zeros(num_gen, dtype=int)
    time_to_event = -MTTF * np.log(np.random.rand(num_gen))

    total_lol_hours = 0.0
    total_loee = 0.0
    total_system_cost = 0.0
    current_time = 0.0

    print(f"Starting Simplified SMCS (10 Units)...")
    start_time = datetime.now()

    while current_time < TOTAL_HOURS:
        h_year = int(current_time) % HOURS_PER_YEAR
        month_idx = month_lookup[h_year]
        current_load = Load[h_year]

        # --- DISPATCH LOGIC ---
        up_mask = (state == 0)
        dispatched = 0.0
        hourly_cost = 0.0
        hydro_used = 0.0
        h_cap = HYDRO_MONTHLY_CAP[month_idx]

        # STEP 1: BASE LOAD (COAL) ALWAYS FIRST
        for i in range(num_gen):
            if up_mask[i] and is_coal[i]:
                contribution = min(Gen[i], current_load - dispatched)
                dispatched += contribution
                hourly_cost += contribution * 1000 * UnitCost[i]
            if dispatched >= current_load: break

        # STEP 2: MERIT ORDER (HYDRO & THERMAL)
        for i in range(num_gen):
            if not up_mask[i] or is_coal[i]: continue
            if dispatched >= current_load: break

            needed = current_load - dispatched
            if is_hydro[i]:
                # Resource constraint (Water)
                potential = min(Gen[i], h_cap - hydro_used)
                contribution = min(potential, needed)
                if contribution > 0:
                    dispatched += contribution
                    hydro_used += contribution
                    hourly_cost += contribution * 1000 * UnitCost[i]
            else:
                contribution = min(Gen[i], needed)
                dispatched += contribution
                hourly_cost += contribution * 1000 * UnitCost[i]

        # --- SYSTEM ADVANCEMENT ---
        dt = min(1.0, np.min(time_to_event))

        if dispatched < current_load - 1e-4:
            total_lol_hours += dt
            total_loee += (current_load - dispatched) * dt

        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle Failures/Repairs
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            ref = MTTF[i] if state[i] == 0 else MTTR[i]
            time_to_event[i] = -ref * np.log(np.random.rand())

    # --- RESULTS ---
    print("\n--- RESULTS (10 UNITS) ---")
    print(f"LOLE (Hours/Year): {total_lol_hours / NUM_YEARS:.4f}")
    print(f"LOEE (MWh/Year):   {total_loee / NUM_YEARS:.2f}")
    print(f"Avg Annual Cost:   LKR {total_system_cost / NUM_YEARS:,.2f}")
    print(f"Sim Runtime:       {datetime.now() - start_time}")


if __name__ == "__main__":
    run_smcs()
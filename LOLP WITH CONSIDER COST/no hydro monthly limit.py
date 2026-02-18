import numpy as np
import pandas as pd
from datetime import datetime
import sys

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================================================
NUM_YEARS = 1000  # Start with 100 to test; increase to 30,000 later
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# =========================================================
# 2. INPUT FILES (Ensure these paths are correct for your PC)
# =========================================================
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"


def load_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CSV files...", flush=True)
    try:
        df_gen = pd.read_csv(GEN_DATA_FILE)

        # Merit Order Sorting
        cost_col = 'Unit Cost (LKR/kWh)'
        if cost_col not in df_gen.columns:
            cost_col = df_gen.columns[df_gen.columns.str.contains('Cost', case=False)][0]

        df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

        Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
        MTTF = df_gen['MTTF (hours)'].values.astype(float)
        MTTR = df_gen['MTTR (hours)'].values.astype(float)
        UnitCost = df_gen[cost_col].values.astype(float)

        df_load = pd.read_csv(LOAD_DATA_FILE)
        Annual_Load = df_load.iloc[:, 0].values.astype(float)

        print(f"Successfully loaded {len(Gen)} generators.", flush=True)
        print(f"Total installed capacity: {np.sum(Gen):,.2f} MW", flush=True)
        print(f"Peak demand in load curve: {np.max(Annual_Load):,.2f} MW", flush=True)

        return Gen, MTTF, MTTR, UnitCost, Annual_Load
    except Exception as e:
        print(f"ERROR LOADING DATA: {e}", flush=True)
        sys.exit(1)


def run_smcs(Gen, MTTF, MTTR, UnitCost, Load):
    n_gen = len(Gen)
    state = np.zeros(n_gen, dtype=int)
    time_to_event = -MTTF * np.log(np.random.rand(n_gen))

    total_LOL_hours = 0.0
    total_LOEE = 0.0
    total_system_cost = 0.0
    current_time = 0.0

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Simulation for {NUM_YEARS} years...", flush=True)
    start_clock = datetime.now()

    # We will log progress every 10 years
    log_interval = HOURS_PER_YEAR * 10

    while current_time < TOTAL_HOURS:
        hour_of_year = int(current_time) % HOURS_PER_YEAR
        current_load = Load[hour_of_year]

        # Merit Order Dispatch
        up_mask = (state == 0)
        dispatched_power = 0.0
        hourly_cost = 0.0

        for i in range(n_gen):
            if not up_mask[i]: continue
            if dispatched_power >= current_load: break

            needed = current_load - dispatched_power
            contribution = min(Gen[i], needed)

            dispatched_power += contribution
            hourly_cost += contribution * 1000 * UnitCost[i]

        # Determine time step
        dt = min(1.0, np.min(time_to_event))

        # Reliability Check
        if dispatched_power < current_load - 1e-4:
            deficit = current_load - dispatched_power
            total_LOL_hours += dt
            total_LOEE += deficit * dt

        total_system_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle State Transitions
        events = np.where(time_to_event <= 1e-6)[0]
        for i in events:
            state[i] = 1 - state[i]
            if state[i] == 0:
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # Progress Log (Every 10 years)
        if int(current_time) % log_interval < dt and current_time > 0:
            pct = (current_time / TOTAL_HOURS) * 100
            print(f" > Progress: Year {int(current_time // HOURS_PER_YEAR)} ({pct:.1f}%)", flush=True)

    # Calculate Results
    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS
    LOEE = total_LOEE / NUM_YEARS
    avg_annual_cost = total_system_cost / NUM_YEARS

    print("\n" + "=" * 40, flush=True)
    print("         FINAL RESULTS (UNCONSTRAINED)", flush=True)
    print("=" * 40, flush=True)
    print(f"LOLE (Hours/Year)      : {LOLE:.4f}", flush=True)
    print(f"LOLP                   : {LOLP:.8f}", flush=True)
    print(f"LOEE (MWh/Year)        : {LOEE:.2f}", flush=True)
    print(f"Avg Annual Cost (LKR)  : {avg_annual_cost:,.2f}", flush=True)
    print(f"Total Simulation Time  : {datetime.now() - start_clock}", flush=True)
    print("=" * 40, flush=True)


if __name__ == "__main__":
    Gen, MTTF, MTTR, UnitCost, Load = load_data()
    run_smcs(Gen, MTTF, MTTR, UnitCost, Load)
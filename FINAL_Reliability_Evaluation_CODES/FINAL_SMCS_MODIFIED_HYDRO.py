import numpy as np
import pandas as pd
import time
import os
import concurrent.futures
from datetime import datetime
import multiprocessing


# 1. RESEARCH CONFIGURATION

TOTAL_RESEARCH_YEARS = 30000
HOURS_PER_YEAR = 8760
NUM_CORES = 4  # Set to 4, 6, or 8 based on your computer

# --- INPUT FILES ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# --- HYDRO GENERATION MATRIX (12 Months x 24 Hours = 288 Values) ---

# Format: [Hour 0, Hour 1, Hour 2, ..., Hour 23]
HYDRO_MATRIX = np.array([
    [542.2903226, 542.2903226, 333.1935484, 596.8387097, 675.1290323, 814.7096774, 801.7096774, 705.0645161, 507.4193548,
    545, 625.1612903, 917.1612903],
    [ 484.3225806, 484.3225806, 277.6129032, 510.3870968, 600.74, 745.516129, 727.0322581, 650.516129, 447.9354839,
    469.2903226, 568.8709677, 870.7419355],
   [  452.8387097, 452.8387097, 242.8064516, 457.5483871, 554.03, 700.516129, 672.8709677, 606.0322581, 412.9677419,
    419.5806452, 522.8387097, 832.8387097],
   [  445.6451613, 445.6451613, 232.3548387, 433.3870968, 537.23, 681.2903226, 660.8709677, 566.8709677,
    388.3225806, 406.5483871, 500.3548387, 807.7741935],
    [ 576.4516129, 576.4516129, 316.3548387, 543.4516129, 661.71, 804.1290323, 802.6451613, 677.483871, 500.5806452,
    541.6451613, 585.4516129, 839.0322581],
    [ 835.9354839, 835.9354839, 523.2258065, 795.9032258, 936.58, 1085.806452, 1092.290323, 922.2580645,
    769.3870968, 838.3870968, 801.9677419, 1003.516129],
    [ 917.3225806, 917.3225806, 613.3870968, 835.4193548, 970.10, 1107.612903, 1132.064516, 1004.419355,
    813.8709677, 876.8709677, 901.8709677, 1160.516129],
   [ 828.5483871, 828.5483871, 513.8387097, 641.0322581, 763.45, 890.9354839, 892.7419355, 836.1935484,
    577.2903226, 615.9354839, 791.6451613, 1079.548387],
    [ 750.0322581, 750.0322581, 399.9354839, 536.8709677, 692.32, 798.2258065, 793.6129032, 703.1290323,
    495.8709677, 482.6774194, 650.9032258, 932.2258065],
    [ 720.1290323, 720.1290323, 355.8064516, 533.3548387, 676.00, 773.4193548, 765.0967742, 654.8709677,
    466.4193548, 468.1935484, 590.7096774, 807],
    [ 630.0967742, 630.0967742, 311.0967742, 519.3225806, 619.81, 665.0967742, 666.3225806, 569.8387097,
    411.4516129, 415.1612903, 527.7419355, 689.516129],
   [604.7419355, 604.7419355, 340.4516129, 524, 626.61, 632.4193548, 623.0967742, 517.1612903, 428.6451613,
    452.0322581, 574.1935484, 711.9032258],
    [ 606.6129032, 606.6129032, 330.0645161, 531.3870968, 631.32, 622.4516129, 619.3225806, 517.9032258,
    437.3870968, 453.8387097, 581.3548387, 717.5483871],
    [582.4193548, 582.4193548, 309.1290323, 509.2903226, 591.42, 595.7096774, 591.6129032, 502.2580645,
    400.5483871, 468.1290323, 551.6774194, 699.2258065],
    [651.3548387, 651.3548387, 388.2903226, 617.7096774, 707.77, 716.1935484, 707.0322581, 603.9354839,
    486.6451613, 576.1612903, 663.1290323, 773.5806452],
    [745.7419355, 745.7419355, 517.3548387, 748.9354839, 795.42, 857.516129, 842.0645161, 729.2903226,
    612.4516129, 744.7419355, 796.9354839, 913.4193548],
   [  854.483871, 854.483871, 678.3548387, 834.9354839, 884.42, 957.2258065, 1000.83871, 894.5806452, 752.4193548,
    904.8064516, 908.6129032, 1036.419355],
    [ 922.8064516, 922.8064516, 781.1935484, 859.2258065, 941.39, 968.0645161, 1049.870968, 984.5806452,
    827.9032258, 1038.290323, 1024.354839, 1122.903226],
    [ 1190.129032, 1190.129032, 1058.032258, 1139.612903, 1225.10, 1189.354839, 1271.741935, 1274.225806,
    1103.677419, 1224.774194, 1167.290323, 1303],
    [ 1191.516129, 1191.516129, 1089.225806, 1150.903226, 1251.10, 1247.741935, 1341.032258, 1310.129032,
    1086.419355, 1180.516129, 1112.032258, 1288.806452],
   [  1071.516129, 1071.516129, 972.1935484, 1073.548387, 1157.61, 1166.16129, 1260.451613, 1200, 968.0322581,
    1053.516129, 1021.483871, 1188.516129],
    [ 914.2903226, 914.2903226, 796.483871, 994, 1073.06, 1075.322581, 1127.709677, 1077.870968, 852.8709677,
    917.0645161, 932.3225806, 1115.516129],
   [  776.1290323, 776.1290323, 599.9032258, 855.6129032, 935.35, 978.7096774, 997.2580645, 922.0967742,
    715.2580645, 765.9677419, 826.2258065, 1035.806452],
    [651.5806452, 651.5806452, 434.2903226, 713.6129032, 800.58, 895.9677419, 877, 790.4193548, 595.5806452,
    635.6451613, 720.516129, 981.2258065],
])



# 2. DATA PREPARATION

def load_data():
    # Load Generators
    df_gen = pd.read_csv(GEN_DATA_FILE)
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)
    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_col].values.astype(float)

    # Load 8760 System Load Curve
    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    # Month mapping for 8760 hours
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = np.array([m for m, days in enumerate(month_days) for _ in range(days * 24)])

    return (Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, month_lookup)


# 3. PARALLEL WORKER ENGINE

def worker_sim(worker_id, years_to_run, data_tuple):
    # Unpack the tuple including the 8760 Load
    Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Load, month_lookup = data_tuple

    # Independent seeding for parallel universes
    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)

    num_gen = len(Gen)
    state = np.zeros(num_gen, dtype=int)

    # Safely initialize time to first event
    rand_vals = np.clip(np.random.rand(num_gen), 1e-9, 0.999999)
    time_to_event = -MTTF * np.log(rand_vals)

    # Aggregators
    total_lol_h, total_loee, total_cost, total_events = 0.0, 0.0, 0.0, 0
    was_in_lol = False
    current_time = 0.0
    target_hours = years_to_run * HOURS_PER_YEAR

    while current_time < target_hours:
        # Time variables
        h_year = int(current_time) % HOURS_PER_YEAR
        month_idx = month_lookup[h_year]
        hour_of_day = int(current_time) % 24

        # 1. GET TARGETS
        current_load = Load[h_year]  # Standard 8760 Hourly Load
        target_hydro_mw = HYDRO_MATRIX[month_idx][hour_of_day]  # Specific Hydro Limit for this Month and Hour

        # Dispatch Variables
        up_mask = (state == 0)
        dispatched = 0.0
        hourly_cost = 0.0
        hydro_dispatched_total = 0.0

        # PASS 1: COAL (BASE LOAD)
        for i in range(num_gen):
            if up_mask[i] and is_coal[i]:
                contrib = min(Gen[i], current_load - dispatched)
                dispatched += contrib
                hourly_cost += contrib * 1000 * UnitCost[i]
            if dispatched >= current_load: break

        # PASS 2: HYDRO & THERMAL (MERIT ORDER)
        for i in range(num_gen):
            if not up_mask[i] or is_coal[i]: continue
            if dispatched >= current_load: break

            needed = current_load - dispatched

            if is_hydro[i]:
                # The remaining hydro allowance for this specific hour
                hydro_allowed_this_hour = max(0.0, target_hydro_mw - hydro_dispatched_total)

                # Hydro is capped by its physical capacity, the grid need, AND the 288-profile limit
                contrib = min(Gen[i], hydro_allowed_this_hour, needed)
                if contrib > 0:
                    dispatched += contrib
                    hydro_dispatched_total += contrib
                    hourly_cost += contrib * 1000 * UnitCost[i]
            else:
                # Regular Thermal
                contrib = min(Gen[i], needed)
                dispatched += contrib
                hourly_cost += contrib * 1000 * UnitCost[i]

        # SYSTEM TIME ADVANCEMENT
        dt = max(min(1.0, np.min(time_to_event)), 1e-4)

        is_failing = dispatched < current_load - 1e-4
        if is_failing:
            total_lol_h += dt
            total_loee += (current_load - dispatched) * dt
            if not was_in_lol: total_events += 1

        was_in_lol = is_failing
        total_cost += hourly_cost * dt
        current_time += dt
        time_to_event -= dt

        # Handle State Transitions (Failures & Repairs)
        expired = np.where(time_to_event <= 1e-6)[0]
        for i in expired:
            state[i] = 1 - state[i]
            ref = MTTF[i] if state[i] == 0 else MTTR[i]
            rand_val = max(1e-9, min(0.999999, np.random.rand()))
            time_to_event[i] = -ref * np.log(rand_val)

    return {"h": total_lol_h, "e": total_loee, "c": total_cost, "f": total_events}



# 4. MASTER CONTROLLER

if __name__ == '__main__':
    print(f"--- Launching Parallel SMCS (288-Value Hourly Hydro Profile) ---")
    data_tuple = load_data()
    start_wall = datetime.now()

    y_per_core = TOTAL_RESEARCH_YEARS // NUM_CORES

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(worker_sim, i, y_per_core, data_tuple) for i in range(NUM_CORES)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Final Aggregation
    agg_h = sum(r["h"] for r in results)
    agg_e = sum(r["e"] for r in results)
    agg_c = sum(r["c"] for r in results)
    agg_f = sum(r["f"] for r in results)

    # Calculate Indices
    LOLE = agg_h / TOTAL_RESEARCH_YEARS
    LOLP = agg_h / (TOTAL_RESEARCH_YEARS * HOURS_PER_YEAR)
    LOEE = agg_e / TOTAL_RESEARCH_YEARS
    LOLF = agg_f / TOTAL_RESEARCH_YEARS
    LOLD = agg_h / agg_f if agg_f > 0 else 0
    avg_annual_cost = agg_c / TOTAL_RESEARCH_YEARS

    # Print Results
    print("\n" + "=" * 40)
    print(f"RESULTS FOR {TOTAL_RESEARCH_YEARS} YEARS")
    print("=" * 40)
    print(f"LOLE : {LOLE:.4f} Hours/Year")
    print(f"LOLP : {LOLP:.8f}")
    print(f"LOEE : {LOEE:.2f} MWh/Year")
    print(f"LOLF : {LOLF:.4f} Occurrences/Year")
    print(f"LOLD : {LOLD:.4f} Hours/Occurrence")
    print(f"Cost : LKR {avg_annual_cost:,.2f} /Year")
    print(f"Total Sim Time: {datetime.now() - start_wall}")
    print("=" * 40)
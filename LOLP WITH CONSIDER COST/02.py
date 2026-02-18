import random
import csv
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
NUM_ITERATIONS = 10000        # for safety during visual printing
GEN_DATA_FILE = "../data/CEB_GEN_FOR_for_each_unit.csv"
LOAD_DATA_FILE = "../data/SriLanka_Load_8760hr_repeat.csv"

# Global variables
Gen = []
FOR = []
Cost = []          # <<< ADDED cost column
Annual_Load_Profile = np.array([])
method = 'NSMCS'

# ============================================================
# 1. LOAD GENERATOR DATA
# ============================================================
def load_generator_data(filepath):
    global Gen, FOR, Cost
    Gen = []
    FOR = []
    Cost = []

    try:
        df_gen = pd.read_csv(filepath)

        # assumes columns: Capacity, FOR, Cost
        Gen = df_gen.iloc[:, 0].astype(float).tolist()
        FOR = df_gen.iloc[:, 1].astype(float).tolist()

        # If cost column exists use it; otherwise assign default cost
        if df_gen.shape[1] >= 3:
            Cost = df_gen.iloc[:, 2].astype(float).tolist()
        else:
            Cost = [15 + i for i in range(len(Gen))]    # default increasing cost

    except Exception as e:
        print("ERROR loading generator data:", e)
        return False

    print(f"Loaded {len(Gen)} generators.")
    return True

# ============================================================
# 2. LOAD ANNUAL DEMAND
# ============================================================
def load_annual_load_profile(filepath):
    global Annual_Load_Profile
    try:
        df_load = pd.read_csv(filepath, header=0)
        Annual_Load_Profile = df_load.iloc[:, 0].astype(float).values

    except Exception as e:
        print("ERROR loading load data:", e)
        return False

    print(f"Loaded {len(Annual_Load_Profile)} hourly load values.")
    return True

# ============================================================
# 3. MONTE CARLO SIMULATION (with Dispatch + Visual Output)
# ============================================================
def run_monte_carlo():

    Gen_np = np.array(Gen)
    FOR_np = np.array(FOR)
    Cost_np = np.array(Cost)
    num_generators = len(Gen)
    num_load_hours = Annual_Load_Profile.size

    H = 0
    N = 0

    print("\n--- STARTING MONTE CARLO WITH DISPATCH ---\n")

    for n in range(NUM_ITERATIONS):
        N += 1

        # -------------------------
        # STEP 1: Availability
        # -------------------------
        rand_vals = np.random.rand(num_generators)
        available_mask = rand_vals > FOR_np    # True = ON
        available_cap = Gen_np * available_mask

        # -------------------------
        # STEP 2: Random Load
        # -------------------------
        load = Annual_Load_Profile[np.random.randint(0, num_load_hours)]

        # -------------------------
        # STEP 3: DISPATCH CHEAPEST
        # -------------------------
        order = np.argsort(Cost_np)   # cheapest generator first
        supply = 0
        selected_units = []
        hour_cost = 0

        for i in order:
            if available_cap[i] == 0:
                continue

            if supply >= load:
                break

            supply += available_cap[i]
            hour_cost += available_cap[i] * Cost_np[i]
            selected_units.append(i)

        # -------------------------
        # STEP 4: Loss of Load check
        # -------------------------
        LOL_flag = load > supply
        if LOL_flag:
            H += 1

        # -------------------------
        # CLEAR VISUAL OUTPUT
        # -------------------------
        if NUM_ITERATIONS <= 10000:     # safe print limit
            print(f"\nIteration {n+1}/{NUM_ITERATIONS}")
            print("Generator ON/OFF:", available_mask.astype(int).tolist())
            print("Available Capacity:", available_cap.tolist())
            print(f"Load = {load:.2f} MW")
            print(f"Selected Units = {selected_units}")
            print(f"Total Supply = {supply:.2f} MW")
            print(f"Hour Cost = {hour_cost:.2f}")
            print("LOSS OF LOAD!" if LOL_flag else "Load Served OK")

        # Still show progress for large N
        if N % 1000000 == 0:
            print(f"{N:,} iterations done... LOLP={H/N:.8f}")

    # -------------------------
    # Final results
    # -------------------------
    LOLP = H / N
    LOLE = LOLP * 8760

    print("\n--- FINAL RESULTS ---")
    print(f"LOLP = {LOLP:.8f}")
    print(f"LOLE = {LOLE:.2f} hours/year")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    if load_generator_data(GEN_DATA_FILE) and load_annual_load_profile(LOAD_DATA_FILE):
        run_monte_carlo()

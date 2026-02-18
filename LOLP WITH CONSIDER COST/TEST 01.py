import random
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple

# --- Configuration ---
NUM_ITERATIONS = 500000  # Reduced for example runtime (use 10M for production)
HOURS_PER_YEAR = 8760
OUTPUT_FILE = "reliability_cost_results.csv"

# --- Placeholder Data (5 Generators) ---
# In a real run, this data must come from your CSV, CEB_Each_unit_Master_data.csv,
# which needs columns: Capacity, FOR, and Cost_per_MWH.

# Units: Gen, FOR, Cost_per_MWH (e.g., $/MWh)
DUMMY_GENERATOR_DATA = [
    [100, 0.02, 10.0],  # Cheap Base Load
    [50, 0.01, 8.0],  # Cheapest
    [200, 0.05, 50.0],  # Expensive Peaker
    [150, 0.03, 15.0],
    [80, 0.04, 30.0]
]

# Load values (Example based on a small peak/off-peak cycle)
DUMMY_LOAD_PROFILE = np.array([100, 150, 200, 300, 450, 350, 180, 120])


def run_monte_carlo_with_cost(gen_data, load_profile, run_label):
    """
    Performs NSMCS, calculates available capacity, and computes the total
    operating cost of all UP generators (no merit order/economic dispatch).
    """

    # 1. Prepare Generator Data Structure (Capacity, FOR, Cost)
    # Convert list of lists into a DataFrame for easy sorting and extraction
    df_gen = pd.DataFrame(gen_data, columns=['Capacity', 'FOR', 'Cost_per_MWH'])

    # Sort units by cost in ASCENDING order (cheapest first)
    # The sorting here is essential for proper economic analysis later,
    # even though we are currently summing all available units.
    df_gen = df_gen.sort_values(by='Cost_per_MWH', ascending=True).reset_index(drop=True)

    Gen_np = df_gen['Capacity'].values

    # --- FIX APPLIED HERE: Corrected dfdf_gen to df_gen ---
    FOR_np = df_gen['FOR'].values
    # --- END FIX ---

    Cost_np = df_gen['Cost_per_MWH'].values  # Cost array, now sorted with Gen_np and FOR_np

    num_generators = len(Gen_np)
    num_load_hours = load_profile.size

    # Cumulative results
    H = 0  # Loss of Load Events
    N = 0  # Total Iterations
    total_operating_cost = 0.0

    print(f"\n--- Starting Cost Simulation: {run_label} ({NUM_ITERATIONS:,} iter) ---")

    # --- CORE LOOP ---
    for n in range(NUM_ITERATIONS):
        N += 1

        # 1. Check Generator Availability
        random_gen_checks = np.random.rand(num_generators)
        outage_mask = (random_gen_checks > FOR_np).astype(int)  # 1 if UP, 0 if DOWN

        # 2. Total Available Capacity
        availableGen = np.sum(Gen_np * outage_mask)

        # 3. Select Load
        load_index = np.random.randint(0, num_load_hours)
        currentLoad = load_profile[load_index]

        # 4. Economic Dispatch & Cost Calculation (Cost of UP generators)

        # Sum Cost_per_MWH * Capacity only for available units
        # This calculates the total cost if ALL available generators were running at full capacity
        available_cost = np.sum(Cost_np * Gen_np * outage_mask)

        total_operating_cost += available_cost

        # 5. Check for Loss of Load
        if currentLoad > availableGen:
            H += 1

    # --- Final Metrics ---
    LOLP = H / N
    LOLE = LOLP * HOURS_PER_YEAR

    # Average Operational Cost (This represents $/MWh * Capacity * Average Time Available)
    # Total cost divided by iterations gives the average cost per simulated hour.
    avg_op_cost_per_hour = total_operating_cost / NUM_ITERATIONS

    print("\n--- Final Results ---")
    print(f"LOLP: {LOLP:.8f}")
    print(f"LOLE: {LOLE:.2f} hrs/yr")
    print(f"Average System Operational Cost per Hour: ${avg_op_cost_per_hour:,.2f}")

    return LOLP, LOLE, avg_op_cost_per_hour


# --- Main Execution ---
if __name__ == "__main__":
    # --- Step 1: Run Base Case ---
    print("\n======== Scenario 1: Base Case Operation ========")
    lolp_base, lole_base, cost_base = run_monte_carlo_with_cost(
        DUMMY_GENERATOR_DATA,
        DUMMY_LOAD_PROFILE,
        "Base_Case"
    )

    # --- Step 2: Run High Outage Case (Cost comparison) ---
    # Increase all FORs by 20% to simulate a bad year
    DUMMY_GENERATOR_DATA_HIGH_FOR = [
        [g[0], min(1.0, g[1] * 1.2), g[2]] for g in DUMMY_GENERATOR_DATA
    ]

    print("\n======== Scenario 2: High Outage (20% Inc FOR) ========")
    lolp_high, lole_high, cost_high = run_monte_carlo_with_cost(
        DUMMY_GENERATOR_DATA_HIGH_FOR,
        DUMMY_LOAD_PROFILE,
        "High_Outage"
    )

    print("\n--- Summary ---")
    print(f"Base Case LOLE: {lole_base:.2f} hrs/yr, Avg Hourly Cost: ${cost_base:,.2f}")
    print(f"High Outage LOLE: {lole_high:.2f} hrs/yr, Avg Hourly Cost: ${cost_high:,.2f}")

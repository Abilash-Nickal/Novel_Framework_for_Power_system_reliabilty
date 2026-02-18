import random
import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------------------------------
# LOAD GENERATOR DATA FROM CSV
# CSV format:
# Gen "unit capacity",FOR "Unit FOR"

gen_data = pd.read_csv("../data/CEB_GEN_FOR_for_each_unit.csv")
gen_data.columns = gen_data.columns.str.strip()   # remove unwanted spaces

Gen = gen_data["unit capacity"].values
FOR = gen_data["Unit FOR"].values
num_gen = len(Gen)

# -------------------------------------------
# LOAD DEMAND CSV
# -------------------------------------------
load = pd.read_csv("../data/SriLanka_Load_8760hr_repeat.csv")

load_values = load.iloc[:, 0].values
load_hours = len(load_values)

# -------------------------------------------
# SIMULATION INITIALIZATION
# -------------------------------------------
H = 0
N = 0
total_iter = 10000000   # 10 million
start_time = datetime.now()
# -------------------------------------------
# MAIN SIMULATION LOOP
# -------------------------------------------
for w in range(total_iter):

    availableGen = 0

    # Check each generator availability (same logic)

    for i in range(num_gen):
        if random.random() > FOR[i]:
            availableGen += Gen[i]

    # Randomly pick a load value (your logic)
    demand = load_values[random.randint(0, load_hours - 1)]

    # Loss-of-load check
    if demand > availableGen:
        H += 1

    N += 1

    # ----------------------
    # PROGRESS PRINT (10%)
    # ----------------------
    if N % 10000 == 0:
            print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")

end_time = datetime.now()
duration = end_time - start_time
# -------------------------------------------
# RESULTS
# -------------------------------------------
LOLP = H / N
LOLE = LOLP * 8760

print("\n----- SMCS RESULTS -----")
print(f"Total Simulated iterations: {total_iter:,}")
print(f"Simulation Runtime: {duration}")
print(f"Total Loss of Load Hours: {H:,.2f}")
print(f"LOLP (Probability): {LOLP:.8f}")
print(f"LOLE (Expected Hours/Year): {LOLE:.2f} hours/year")

# -------------------------------------------
# APPEND RESULTS TO CSV
# -------------------------------------------
output_file = "LOLP_results.csv"

result = pd.DataFrame({
    "LOLP": [LOLP],
    "LossOfLoadHours": [H],
    "TotalHours": [N]
})

# If file does not exist → create with header
if not os.path.exists(output_file):
    result.to_csv(output_file, index=False, mode='w', header=True)

# If file exists → append without header
else:
    result.to_csv(output_file, index=False, mode='a', header=False)

print("Result appended to:", output_file)

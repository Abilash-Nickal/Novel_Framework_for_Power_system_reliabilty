import random
import pandas as pd
import numpy as np

# ------------------------------
#  LOAD GENERATOR DATA FROM CSV
# ------------------------------

gen_data = pd.read_csv("../data/CEB_GEN_Each_unit_Master_data.csv")

Gen = gen_data["Unit Capacity (MW)"].values
MTTR = gen_data["MTTR (hours)"].values
MTTF = gen_data["MTTF (hours)"].values

print(f" Loaded {len(Gen)} generators.")

num_gen = len(Gen)

# ------------------------------
#  LOAD DEMAND (8760 hours)
# ------------------------------

load = pd.read_csv("../")      # single column "Load"
load_values = load["Load (MW)"].values
load_hours = len(load_values)

print(f" Loaded {len(load)} load.")
# ------------------------------
#  INITIALIZE VARIABLES
# ------------------------------

var_In = np.zeros(num_gen)
var_Out = np.zeros(num_gen)
time = np.zeros(num_gen)

H = 0
N = 0
l = 0  # load row index
NUM_ITERATIONS=10000000
# ------------------------------
#  INITIAL CYCLE GENERATION
# ------------------------------

for i in range(num_gen):
    x = np.random.rand()
    var_In[i] = -round(MTTF[i] * np.log(x), 0)

    x = np.random.rand()
    var_Out[i] = -round(MTTR[i] * np.log(x), 0)

    time[i] = var_In[i] + var_Out[i]

# ------------------------------
#  MAIN SIMULATION LOOP
# ------------------------------

for w in range(NUM_ITERATIONS):  # 10 million iterations

    availableGen = 0  # reset generator availability
    for i in range(num_gen):

        # create new up/down cycle if needed
        if time[i] < w:
            x = np.random.rand()
            var_In[i] = -round(MTTF[i] * np.log(x), 0)

            x = np.random.rand()
            var_Out[i] = -round(MTTR[i] * np.log(x), 0)

            time[i] += var_In[i] + var_Out[i]

        # check availability
        if (time[i] - w) >= var_Out[i]:
            availableGen += Gen[i]
    # -----------------------------------
    # update load (8760 hr cycle repeat)
    # -----------------------------------
    demand = load_values[l]
    l += 1
    if l == load_hours:
        l = 0

    # LOLP count
    if availableGen < demand:
        H += 1
    if w % (NUM_ITERATIONS // 10) == 0 and w != 0:
            print(f"{(w / NUM_ITERATIONS) * 100:.0f}% completed...")
    N += 1
    if N % 1000000 == 0:
            print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")


# ------------------------------
#  CALCULATE LOLP
# ------------------------------

LOLP = H / N
LOLE = LOLP * 8760

print(f"Final LOLP (Probability): {LOLP:.10f}")
print(f"Final LOLP (Probability): {LOLE:.10f}")

# save results to CSV
import os

output_file = "SMCS_result_LoLP.csv"

# Create new result row
result = pd.DataFrame({
    "LOLP": [LOLP],
    "LossOfLoadHours": [H],
    "TotalHours": [N]
})

# If file does NOT exist → create with header
if not os.path.exists(output_file):
    result.to_csv(output_file, index=False, mode='w', header=True)

# If file exists → append without header
else:
    result.to_csv(output_file, index=False, mode='a', header=False)

print("Result appended to", output_file)

print("Saved: SMCS_result_LoLP.csv")

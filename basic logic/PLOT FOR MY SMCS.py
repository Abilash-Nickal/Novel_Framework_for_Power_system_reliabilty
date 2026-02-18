import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
#  LOAD GENERATOR DATA
# ------------------------------

gen_data = pd.read_csv("../data/CEB_GEN_MTTR_&_MTTF_for_each_unit.csv")

Gen = gen_data["Capacity (MW)"].values
MTTR = gen_data["MTTR (hours)"].values
MTTF = gen_data["MTTF (hours)"].values

num_gen = len(Gen)
print(f" Loaded {num_gen} generators.")

# ------------------------------
#  LOAD DEMAND (8760 hours)
# ------------------------------

load = pd.read_csv("../data/SriLanka_Load_8760hr_repeat.csv")
load_values = load["Load (MW)"].values
load_hours = len(load_values)

print(f" Loaded {len(load)} load hours.")

# ------------------------------
#  INITIALIZE VARIABLES
# ------------------------------

var_In = np.zeros(num_gen)
var_Out = np.zeros(num_gen)
time = np.zeros(num_gen)

H = 0
N = 0
l = 0

NUM_ITERATIONS = 1000000

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
#  GENERATOR STATE STORAGE (FOR PLOT)
# ------------------------------

# Store only first 20000 iterations → avoid memory issues
PLOT_LIMIT = 20000

gen_state_history = np.zeros((num_gen, PLOT_LIMIT))

# ------------------------------
#  MAIN SIMULATION LOOP
# ------------------------------

for w in range(NUM_ITERATIONS):

    availableGen = 0

    for i in range(num_gen):

        # refresh cycles
        if time[i] < w:
            x = np.random.rand()
            var_In[i] = -round(MTTF[i] * np.log(x), 0)

            x = np.random.rand()
            var_Out[i] = -round(MTTR[i] * np.log(x), 0)

            time[i] += var_In[i] + var_Out[i]

        # generator is UP if remaining time > repair period
        is_up = (time[i] - w) >= var_Out[i]
        if is_up:
            availableGen += Gen[i]

        # store generator state (only first 20,000 iterations)
        if w < PLOT_LIMIT:
            gen_state_history[i, w] = 0 if is_up else 1

    # rotating load
    demand = load_values[l]
    l += 1
    if l == load_hours:
        l = 0

    # LOLP counter
    if availableGen < demand:
        H += 1

    N += 1

    if w % (NUM_ITERATIONS // 10) == 0 and w != 0:
        print(f"{(w / NUM_ITERATIONS) * 100:.0f}% completed...")

    if N % 1000000 == 0:
        print(f"Progress: {N:,} iterations completed. Current LOLP: {H / N:.10f}")

# ------------------------------
#  RESULTS
# ------------------------------

LOLP = H / N
LOLE = LOLP * 8760

print(f"Final LOLP: {LOLP:.10f}")
print(f"Final LOLE (hr/yr): {LOLE:.2f}")

# ------------------------------
#  PLOT GENERATOR STATES
# ------------------------------

plt.figure(figsize=(15, 6))

for i in range(num_gen):
    plt.step(range(PLOT_LIMIT), gen_state_history[i], where='post', label=f"Gen {i+1}")

plt.yticks([0, 1], ["UP", "DOWN"])
plt.xlabel("Iterations (first 20,000 only)")
plt.ylabel("State")
plt.title("Generator State vs Iteration (NMSC) – First 20,000 Samples")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# SAVE RESULTS
# ------------------------------

output_file = "result_LoLP.csv"
result = pd.DataFrame({
    "LOLP": [LOLP],
    "LossOfLoadHours": [H],
    "TotalHours": [N]
})

import os
if not os.path.exists(output_file):
    result.to_csv(output_file, index=False, mode='w', header=True)
else:
    result.to_csv(output_file, index=False, mode='a', header=False)

print("Saved:", output_file)

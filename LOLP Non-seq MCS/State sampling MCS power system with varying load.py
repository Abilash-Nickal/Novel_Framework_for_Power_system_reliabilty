import random
import pandas as pd
import numpy as np
# Generator capacities and forced outage rates
Gen = [100, 200]
FOR = [0.01, 0.02]

# Load levels for different time blocks
load = pd.read_csv("../basic logic/load_rbts.csv")

H = 0
N = 0

for _ in range(10000000):
    availableGen = 0

    # Check if each generator is available
    if random.random() > FOR[0]:
        availableGen += Gen[0]
    if random.random() > FOR[1]:
        availableGen += Gen[1]

    # Randomly select a load level based on time block probabilities
    load_rand =random.randint(0, 8735)
    demand = load.iloc[load_rand, 0]
    # Check if load exceeds available generation
    if  demand  > availableGen:
        H += 1
    N += 1

# Calculate LOLP
LOLP = H / N
print("LOLP:", LOLP)

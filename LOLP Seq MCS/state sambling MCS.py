import random
import pandas as pd
import numpy as np


# Generator capacities and forced outage rates
Gen = [100, 200]

MTTF=[4380.0, 2190.0]
MTTR=[44.24, 44.69]

# Load levels for different time blocks
load = pd.read_csv("../data/SriLanka_Load_8760hr_repeat.csv")

var_In = np.zeros(2)
var_Out = np.zeros(2)
time = np.zeros(2)
l=0
availableGen = 0
H = 0
N = 0

for i in range(2):
    x = np.random.rand()
    var_In[i] = -round(MTTF[i] * np.log(x), 0)
    x = np.random.rand()
    var_Out[i] = -round(MTTR[i] * np.log(x), 0)
    time[i] = var_In[i] + var_Out[i]

for w in range(10000000):
    for i in range(2):
        if time[i] < w:
            x = np.random.rand()
            var_In[i] = -round(MTTF[i] * np.log(x), 0)
            x = np.random.rand()
            var_Out[i] = -round(MTTR[i] * np.log(x), 0)
            time[i] += var_In[i] + var_Out[i]
            if (time[i] - w) >= var_Out[i]:
                availableGen += Gen[i]
    demand=load.iloc[l,0]
    if w==8735:
        l=0
    if availableGen < demand:
        H=H+1
    N=N+1
# Calculate LOLP
LOLP = H / N
print("LOLP:", LOLP)

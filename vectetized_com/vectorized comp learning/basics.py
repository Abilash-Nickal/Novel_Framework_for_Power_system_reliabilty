import numpy as np

# -----------------------------
# 1. DUMMY SYSTEM DATA
# -----------------------------

hours = 24                      # simulate 24 hours instead of 8760
num_gen = 3                     # 3 generators

# Generator capacities (MW)
Gen = np.array([100, 80, 50])

# Generator cost (merit order)
UnitCost = np.array([5, 8, 12])

# Hourly load (MW)
Load = np.array([
120,130,140,150,160,170,
180,190,200,210,220,230,
240,230,220,210,200,190,
180,170,160,150,140,130
])

# -----------------------------
# 2. GENERATOR STATES
# -----------------------------
# 1 = UP, 0 = DOWN

States = np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Gen1 always up
[1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Gen2 fails 2 hours
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]   # Gen3 always up
])

# -----------------------------
# 3. VECTORZIED AVAILABILITY
# -----------------------------

Avail = States * Gen[:,None]

print("Generator availability matrix (MW):")
print(Avail)

# -----------------------------
# 4. CUMULATIVE CAPACITY
# -----------------------------

cum_Avail = np.cumsum(Avail, axis=0)

print("\nCumulative available generation:")
print(cum_Avail)

# -----------------------------
# 5. MERIT ORDER DISPATCH
# -----------------------------

cum_Avail_prev = np.zeros_like(cum_Avail)
cum_Avail_prev[1:,:] = cum_Avail[:-1,:]

Dispatched = np.minimum(
    Avail,
    np.maximum(0, Load - cum_Avail_prev)
)

print("\nDispatched generation:")
print(Dispatched)

# -----------------------------
# 6. TOTAL SYSTEM GENERATION
# -----------------------------

TotalGen = np.sum(Dispatched, axis=0)

print("\nTotal generation each hour:")
print(TotalGen)

# -----------------------------
# 7. SHORTFALL CALCULATION
# -----------------------------

Shortfall = np.maximum(0, Load - TotalGen)

print("\nLoad:")
print(Load)

print("\nShortfall:")
print(Shortfall)

# -----------------------------
# 8. RELIABILITY INDICES
# -----------------------------

LOL_hours = np.sum(Shortfall > 0)
LOEE = np.sum(Shortfall)

print("\nResults")
print("LOL Hours:", LOL_hours)
print("LOEE:", LOEE)
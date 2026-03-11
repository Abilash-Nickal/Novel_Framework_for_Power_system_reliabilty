import numpy as np

# --------------------------------
# 1. SYSTEM DATA (Dummy Example)
# --------------------------------

hours = 48                 # simulate 48 hours
num_gen = 4

Gen = np.array([300,200,150,100])     # MW
MTTF = np.array([500,600,700,800])
MTTR = np.array([20,30,40,50])

is_coal  = np.array([1,0,0,0])
is_hydro = np.array([0,1,0,0])

UnitCost = np.array([4,5,8,12])       # LKR/kWh

Load = np.random.randint(400,700,hours)

# Monthly hydro capacity
HYDRO_CAP = 250

# --------------------------------
# 2. GENERATE FAILURE TRANSITIONS
# --------------------------------

transitions_list = []

for i in range(num_gen):

    num_cycles = 200

    ttf = -MTTF[i] * np.log(np.random.rand(num_cycles))
    ttr = -MTTR[i] * np.log(np.random.rand(num_cycles))

    durations = np.empty(num_cycles*2)

    durations[0::2] = ttf
    durations[1::2] = ttr

    transitions = np.cumsum(durations)

    transitions_list.append(transitions)

# --------------------------------
# 3. GENERATOR STATES MATRIX
# --------------------------------

hours_vec = np.arange(hours)

States = np.zeros((num_gen,hours))

for i in range(num_gen):

    idx = np.searchsorted(transitions_list[i], hours_vec)

    States[i] = (idx % 2 == 0)

# --------------------------------
# 4. AVAILABILITY MATRIX
# --------------------------------

Avail = States * Gen[:,None]

# --------------------------------
# 5. HYDRO ENERGY LIMIT
# --------------------------------

hydro_avail = Avail[is_hydro==1]

cum_hydro = np.cumsum(hydro_avail,axis=0)

cum_hydro_cap = np.minimum(cum_hydro,HYDRO_CAP)

hydro_dispatch = np.zeros_like(hydro_avail)

hydro_dispatch[0] = cum_hydro_cap[0]

if len(hydro_avail)>1:
    hydro_dispatch[1:] = cum_hydro_cap[1:] - cum_hydro_cap[:-1]

Avail[is_hydro==1] = hydro_dispatch

# --------------------------------
# 6. MERIT ORDER DISPATCH
# --------------------------------

cum_Avail = np.cumsum(Avail,axis=0)

cum_prev = np.zeros_like(cum_Avail)
cum_prev[1:] = cum_Avail[:-1]

Dispatched = np.minimum(
    Avail,
    np.maximum(0,Load - cum_prev)
)

# --------------------------------
# 7. SYSTEM GENERATION
# --------------------------------

TotalGen = cum_Avail[-1]

Shortfall = np.maximum(0,Load-TotalGen)

# --------------------------------
# 8. RELIABILITY INDICES
# --------------------------------

LOLP = np.mean(Shortfall>0)

LOLE = np.sum(Shortfall>0)

LOEE = np.sum(Shortfall)

# --------------------------------
# 9. SYSTEM COST
# --------------------------------

hourly_cost = Dispatched * UnitCost[:,None] * 1000

total_cost = np.sum(hourly_cost)

print("LOLP:",LOLP)
print("LOLE:",LOLE)
print("LOEE:",LOEE)
print("Total Cost:",total_cost)
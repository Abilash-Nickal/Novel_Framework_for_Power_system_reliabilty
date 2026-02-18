import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# EXAMPLE SYSTEM (modify as needed)
# -----------------------------
Gen = [100, 200]               # MW
MTTF = [4380, 2190]            # hours
MTTR = [44, 44]                # hours
HOURS = 10000            # simulation duration

num_gens = len(Gen)

# -----------------------------
# STATE TRACKING ARRAYS
# -----------------------------
state = np.zeros(num_gens, dtype=int)     # 0 = UP, 1 = DOWN
time_to_event = -np.array(MTTF) * np.log(np.random.rand(num_gens))

# To store states for plotting
state_history = np.zeros((num_gens, HOURS))

# -----------------------------
# SEQUENTIAL SIMULATION
# -----------------------------
t = 0
while t < HOURS:

    # Record current states
    for g in range(num_gens):
        state_history[g, t] = state[g]

    # Find next event time
    dt = min(1, np.min(time_to_event))   # cap at 1 hour step
    time_to_event -= dt
    t += 1

    # Check events
    events = np.where(time_to_event <= 0)[0]
    for g in events:
        # Flip state
        state[g] = 1 - state[g]

        # Resample next event time
        if state[g] == 0:     # repaired → next event is failure
            time_to_event[g] = -MTTF[g] * np.log(np.random.rand())
        else:                 # failed → next event is repair
            time_to_event[g] = -MTTR[g] * np.log(np.random.rand())

# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(12,6))

for g in range(num_gens):
    plt.step(range(HOURS), state_history[g], where='post', label=f"Generator {g+1}")

plt.yticks([0, 1], ["UP", "DOWN"])
plt.xlabel("Time (hours)")
plt.ylabel("State")
plt.title("Generator State vs Time (SMCS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

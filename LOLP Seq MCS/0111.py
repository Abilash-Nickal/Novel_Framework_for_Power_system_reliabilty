import numpy as np
import pandas as pd


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(gen_file, load_file):
    df_gen = pd.read_csv(gen_file)
    Gen = df_gen['Unit Capacity (MW)'].values
    MTTF = df_gen['MTTF (hours)'].values
    MTTR = df_gen['MTTR (hours)'].values

    df_load = pd.read_csv(load_file)
    Load = df_load.iloc[:, 0].values.astype(float)

    # Ensure 8760-hour profile
    if len(Load) < 8760:
        Load = np.tile(Load, int(np.ceil(8760 / len(Load))))[:8760]
    else:
        Load = Load[:8760]

    return Gen, MTTF, MTTR, Load


# -----------------------------
# 2. Sequential Monte Carlo Simulation
# -----------------------------
def run_smcs(Gen, MTTF, MTTR, Load, years=100):
    HOURS = 1000
    total_hours = HOURS * years

    n = len(Gen)

    # All generators start UP (0 = UP, 1 = DOWN)
    state = np.zeros(n, dtype=int)

    # Initial TTF for all generators
    next_event = -MTTF * np.log(np.random.rand(n))

    LOL_hours = 0
    t = 0

    while t < total_hours:

        # Available capacity
        available = np.sum(Gen[state == 0])

        # Current hourly demand
        load_now = Load[int(t) % 8760]

        # Time until next failure/repair event
        dt = min(1.0, np.min(next_event))

        # If load > supply, add outage time
        if available < load_now:
            LOL_hours += dt

        # Advance time
        t += dt
        next_event -= dt

        # Handle generators whose event happened (failure or repair)
        events = np.where(next_event <= 0)[0]
        for i in events:
            state[i] = 1 - state[i]  # flip state

            # Assign next event time
            if state[i] == 0:
                next_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                next_event[i] = -MTTR[i] * np.log(np.random.rand())

    # Results
    LOLP = LOL_hours / total_hours
    LOLE = LOL_hours / years

    print("----- RESULTS -----")
    print(f"Total LOL Hours: {LOL_hours:.2f}")
    print(f"LOLP: {LOLP:.8f}")
    print(f"LOLE: {LOLE:.2f} hours/year")


# -----------------------------
# 3. Main
# -----------------------------
if __name__ == "__main__":
    Gen, MTTF, MTTR, Load = load_data(
        "../DATA/CEB_GEN_Each_unit_Master_data.csv",
        "../data/SriLanka_Load_8760hr_repeat.csv"
    )

    run_smcs(Gen, MTTF, MTTR, Load, years=10000)

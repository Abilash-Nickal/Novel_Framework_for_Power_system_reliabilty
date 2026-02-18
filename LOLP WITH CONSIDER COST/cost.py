import numpy as np

# -----------------------------------------
# GENERATOR DATA (10 units)
# -----------------------------------------
gen_cap = np.array([100, 120, 80, 150, 200, 50, 60, 70, 90, 110])
FOR     = np.array([0.05, 0.1, 0.03, 0.07, 0.08, 0.02, 0.04, 0.05, 0.06, 0.03])
cost    = np.array([15, 18, 20, 22, 13, 25, 30, 28, 19, 16])

num_gen = len(gen_cap)

# -----------------------------------------
# DEMAND PROFILE (Example 10 hours)
# -----------------------------------------
load = np.array([400, 500, 600, 450, 300, 700, 650, 550, 500, 480])
hours = len(load)

# -----------------------------------------
# RESULT VARIABLES
# -----------------------------------------
LOLP = 0
total_cost = 0

# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
print("\n===================== SMCS SIMULATION START =====================\n")

for h in range(hours):

    print(f"\n-------------------- Hour {h+1} --------------------")

    # ==============================
    # STEP 1: CHECK AVAILABILITY
    # ==============================
    rand = np.random.rand(num_gen)
    state = (rand > FOR).astype(int)          # 1 = available, 0 = failed
    available_cap = gen_cap * state

    print("\nGenerator States (1 = ON, 0 = OFF)")
    print(state)

    # ==============================
    # STEP 2: READ DEMAND
    # ==============================
    demand = load[h]
    print(f"\nDemand = {demand} MW")

    # ==============================
    # STEP 3: DISPATCH LOWEST COST UNITS
    # ==============================
    order = np.argsort(cost)                  # cheapest first

    supply = 0
    hour_cost = 0
    selected_units = []

    for i in order:
        if available_cap[i] == 0:
            continue
        if supply >= demand:
            break

        supply += available_cap[i]
        hour_cost += available_cap[i] * cost[i]
        selected_units.append(i)

    print(f"\nSelected Units (cheapest first): {selected_units}")
    print(f"Individual Capacities Used: {[available_cap[i] for i in selected_units]}")
    print(f"Total Supply = {supply} MW")

    # ==============================
    # LOLP CHECK
    # ==============================
    if supply < demand:
        LOLP += 1
        print("⚠  LOSS OF LOAD! Supply < Demand")
    else:
        print("✓ Load served successfully.")

    total_cost += hour_cost

    print(f"Cost for Hour {h+1} = {hour_cost} LKR\n")

# -----------------------------------------
# FINAL RESULTS
# -----------------------------------------
print("\n===================== SMCS SIMULATION END =====================\n")
print(f"Total Cost = {total_cost} LKR")
print(f"LOLP = {LOLP} / {hours} = {LOLP/hours:.5f}")

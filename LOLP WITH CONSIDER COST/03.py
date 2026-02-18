import numpy as np
import random

# -----------------------------------------------------------
# DUMMY GENERATOR DATA (10 GENERATORS)
# -----------------------------------------------------------
Gen = np.array([100, 120, 80, 60, 150, 200, 90, 110, 130, 140])   # Capacity MW
FOR = np.array([0.05, 0.08, 0.03, 0.10, 0.07, 0.02, 0.06, 0.05, 0.04, 0.03])  # Forced outage rate
Cost = np.array([5, 6, 7, 8, 5, 9, 6, 7, 5, 6])    # Cost per MW (dummy)

num_gen = len(Gen)

# -----------------------------------------------------------
# DUMMY LOAD PROFILE (varies each iteration)
# -----------------------------------------------------------
Load_profile = np.array([  300, 320, 350, 380, 420, 480, 520, 550, 580, 600,
    620, 630, 640, 650, 660, 670, 700, 730, 760, 740,
    700, 650, 500, 350,  300, 320, 350, 380, 420, 480, 520, 550, 580, 600,
    620, 630, 640, 650, 660, 670, 700, 730, 760, 740,
    700, 650, 500, 350])

# -----------------------------------------------------------
# SIMULATION SETTINGS
# -----------------------------------------------------------
iterations = 1000000     # Keep small for visibility
LOLP = 0
loss_count = 0

print("\n=========== SMCS START ===========\n")

# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
for it in range(iterations):

    # STEP 1: Choose random load value from profile
    load = random.choice(Load_profile)

    # STEP 2: Determine generator availability (1=up, 0=down)
    availability = (np.random.rand(num_gen) > FOR).astype(int)

    # Step 3: Available capacity
    available_capacities = Gen * availability

    # Step 4: Cost-based dispatch
    # sort generators by cost (low → high)
    order = np.argsort(Cost)
    dispatched_gen = np.zeros(num_gen)
    remaining_load = load

    for idx in order:
        if availability[idx] == 1:   # only if generator is up
            gen_capacity = Gen[idx]
            dispatch = min(gen_capacity, remaining_load)
            dispatched_gen[idx] = dispatch
            remaining_load -= dispatch

            if remaining_load <= 0:
                break

    # Step 5: Check if loss of load occurred
    if remaining_load > 0:
        loss = "YES"
        loss_count += 1
    else:
        loss = "NO"

    # -------------------------------------------------------
    # PRINT CLEAR VISUAL OUTPUT
    # -------------------------------------------------------
    print(f"Iteration {it+1}")
    print(f"Load = {load} MW")
    print("Generator availability (1=UP, 0=DOWN):")
    print(availability)
    print("Dispatched MW:")
    print(dispatched_gen)
    print(f"Remaining load not served = {remaining_load} MW")
    print(f"Loss of Load = {loss}")
    print("------------------------------------------\n")

# -----------------------------------------------------------
# RESULTS
# -----------------------------------------------------------
LOLP = loss_count / iterations

print("\n=========== RESULTS ===========")
print(f"Total iterations = {iterations}")
print(f"Loss of Load occurrences = {loss_count}")
print(f"LOLP = {LOLP:.4f}")
print("================================\n")

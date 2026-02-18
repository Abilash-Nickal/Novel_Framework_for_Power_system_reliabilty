import numpy as np
import random


# DUMMY GENERATOR DATA (10 GENERATORS)
Gen = np.array([100, 120, 80, 60, 150, 200, 90, 110, 130, 140])   # Capacity MW
FOR = np.array([0.05, 0.08, 0.03, 0.10, 0.07, 0.02, 0.06, 0.05, 0.04, 0.03])  # Forced outage rate
Cost = np.array([5, 6, 7, 8, 5, 9, 6, 7, 5, 6])    # Cost per MW (dummy)

num_gen = len(Gen)


# DUMMY LOAD PROFILE (varies each iteration)
Load_profile = np.array([400, 450, 500, 550, 600,  300, 320, 350, 380, 420, 480, 520, 550, 580, 600,
    620, 630, 640, 650, 660, 670, 700, 730, 760, 740,
    700, 650, 500, 350,  300, 320, 350, 380, 420, 480, 520, 550, 580, 600,
    620, 630, 640, 650, 660, 670, 700, 730, 760, 740,
    700, 650, 500, 350])


# SIMULATION SETTINGS
iterations = 2000
loss_count = 0

print("\n=========== SMCS START ===========\n")

for it in range(iterations):

    # STEP 1: random load
    load = random.choice(Load_profile)

    # STEP 2: generator availability (1=up, 0=down)
    availability = (np.random.rand(num_gen) > FOR).astype(int)

    # STEP 3: Select 2 UP generators as spinning reserve (no generation)
    up_indices = np.where(availability == 1)[0]

    if len(up_indices) >= 2:
        spinning_reserve = np.random.choice(up_indices, size=2, replace=False)
    else:
        spinning_reserve = []

    # STEP 4: Cost-based dispatch
    order = np.argsort(Cost)
    dispatched_gen = np.zeros(num_gen)
    remaining_load = load

    for idx in order:

        # If generator is DOWN → skip
        if availability[idx] == 0:
            continue

        # If generator selected as spinning reserve → skip
        if idx in spinning_reserve:
            continue

        # Dispatch normally
        dispatch = min(Gen[idx], remaining_load)
        dispatched_gen[idx] = dispatch
        remaining_load -= dispatch

        if remaining_load <= 0:
            break

    # STEP 5: Check for loss of load
    if remaining_load > 0:
        loss = "YES"
        loss_count += 1
    else:
        loss = "NO"

    # PRINT OUTPUT
    print(f"Iteration {it+1}")
    print(f"Load = {load} MW")
    print("Availability:", availability)
    print("Spinning reserve generators:", spinning_reserve)
    print("Dispatched MW:", dispatched_gen)
    print(f"Remaining unmet load = {remaining_load} MW")
    print(f"Loss of Load = {loss}")
    print("------------------------------------------\n")

# RESULTS
LOLP = loss_count / iterations

print("\n=========== RESULTS ===========")
print(f"Total iterations = {iterations}")
print(f"Loss of Load occurrences = {loss_count}")
print(f"LOLP = {LOLP:.4f}")
print("================================\n")

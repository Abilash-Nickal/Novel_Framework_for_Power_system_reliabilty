import random

# Generator capacities and forced outage rates
Gen = [10, 20]
FOR = [0.01, 0.02]

# Load levels for different time blocks
LoadLevels = [5, 10, 25, 20]

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
    load_rand = random.random()
    if load_rand < 0.25:
        currentLoad = LoadLevels[0]
    elif load_rand < 0.5:
        currentLoad = LoadLevels[1]
    elif load_rand < 0.75:
        currentLoad = LoadLevels[2]
    else:
        currentLoad = LoadLevels[3]

    # Check if load exceeds available generation
    if currentLoad > availableGen:
        H += 1
    N += 1

# Calculate LOLP
LOLP = H / N
print("LOLP:", LOLP)

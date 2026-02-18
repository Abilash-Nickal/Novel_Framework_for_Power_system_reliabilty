import numpy as np

# ---------------------------------------------------------
# CREATE 360 LOAD VALUES (dummy but realistic)
# ---------------------------------------------------------

np.random.seed(1)

# Base daily shape (morning peak + evening peak)
daily_shape = np.array([
    300, 320, 350, 380, 420, 480, 520, 550, 580, 600,
    620, 630, 640, 650, 660, 670, 700, 730, 760, 740,
    700, 650, 500, 350
])

# Scale to represent typical system variation
daily_shape = daily_shape * 1.0    # adjust scale if needed

Load_profile_360 = []

for day in range(360):
    # apply small random variation per day
    variation = np.random.normal(0, 20, 24)
    day_load = daily_shape + variation

    # round and avoid negative
    day_load = np.clip(day_load.astype(int), 200, None)

    Load_profile_360.extend(day_load)

Load_profile_360 = np.array(Load_profile_360)

print("Created 360-day load profile with:")
print("Total hours:", len(Load_profile_360))
print(Load_profile_360)

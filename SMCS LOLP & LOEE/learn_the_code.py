import numpy as np
import pandas as pd
from datetime import datetime

# 1. SIMULATION CONFIGURATION
NUM_YEARS = 100    # Change to 30000 for final results
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR


# 2. INPUT FILES

GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv" # Generator data with costs
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"# Load curve data (hourly for 1 year)


# 3. MONTHLY HYDRO CAPACITY (MW)

HYDRO_MONTHLY_CAP = np.array( [853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118]) # Monthly hydro capacity limits (MW) for each month (Jan to Dec)


# 4. DATA LOADING

def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE) # Load generator data from CSV file

    # ---- Merit Order Sorting (Cheapest First) ----
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)# Sort generators by their unit cost in ascending order (cheapest first)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)# Extract generator capacities (MW) as a numpy array of floats
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)

    # Faster hydro flag
    hydro_generators = (
        df_gen['TYPES'].str.upper().str.strip() == 'HYDRO' # Create a boolean array where True indicates the generator is of type 'HYDRO' (case-insensitive, whitespace-trimmed)
    ).values # Extract the unit cost column as a numpy array of floats

    UnitCost = df_gen[cost_col].values.astype(float) # Extract the load curve data from CSV file, assuming it has one column with hourly load values for a non-leap year (8760 hours)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)# Extract the load curve data from CSV file, assuming it has one column with hourly load values for a non-leap year (8760 hours)

    # ---- Exact Month Lookup (Non-leap year) ----
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days): # Create a month lookup array that maps each   hour of the year to its corresponding month index (0 for Jan, 1 for Feb, ..., 11 for Dec)
        month_lookup.extend([m] * (days * 24))# For each month, extend the month_lookup list with the month index repeated for the number of hours in that month (days * 24)
    month_lookup = np.array(month_lookup) # Convert the month_lookup list to a numpy array for efficient indexing

    return Gen, MTTF, MTTR, hydro_generators, UnitCost, Annual_Load, month_lookup # Return all the loaded and processed data as numpy arrays


# =========================================================
# 5. SEQUENTIAL MONTE CARLO SIMULATION
# =========================================================
def run_smcs(Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup): # Run the Sequential Monte Carlo Simulation (SMCS) to calculate reliability metrics and costs

    num_of_generators = len(Gen)# Get the number of generators from the length of the Gen array

    # 0 = UP, 1 = DOWN
    current_state = np.zeros(num_of_generators, dtype=int) # Initialize the current state of each generator to 0 (UP) using a numpy array of integers
    # 0 means every single generator is working perfectly at Hour 0.
    # Initial time to failure
    time_to_event = -MTTF * np.log(np.random.rand(num_of_generators)) # Sample the initial time to failure for each generator using an exponential distribution based on their MTTF values
    #The math -MTTF * ln(random) creates a realistic, random schedule of breakdowns for the future.

    total_LOL_hours = 0.0
    total_LOEE = 0.0          # MWh
    total_system_cost = 0.0   # LKR

    current_time = 0.0

    print(f"Starting SMCS for {NUM_YEARS} years...") # Print a message indicating the start of the simulation and the number of years being simulated
    start_clock = datetime.now()

    while current_time < TOTAL_HOURS: # Main simulation loop that runs until the current time reaches the total hours for the specified number of years


        hour_of_year = int(current_time) % HOURS_PER_YEAR
        month_idx = month_lookup[hour_of_year]
        current_load = Load[hour_of_year]

        # ---------------------------------------------
        # MERIT ORDER DISPATCH
        # ---------------------------------------------
        up_mask = (current_state == 0)

        dispatched_power = 0.0
        hourly_cost = 0.0
        current_hydro_used = 0.0
        hydro_cap = HYDRO_MONTHLY_CAP[month_idx]

        for i in range(num_of_generators):

            if not up_mask[i]:
                continue

            if dispatched_power >= current_load:
                break

            needed = current_load - dispatched_power

            if is_hydro[i]:
                potential = min(Gen[i], hydro_cap - current_hydro_used)
                contribution = min(potential, needed)

                if contribution > 0:
                    dispatched_power += contribution
                    current_hydro_used += contribution

                    # MW × 1000 × (LKR/kWh) = LKR/hour
                    hourly_cost += contribution * 1000 * UnitCost[i]

            else:
                contribution = min(Gen[i], needed)
                dispatched_power += contribution

                hourly_cost += contribution * 1000 * UnitCost[i]

        # ---------------------------------------------
        # TIME STEP (Event-based)
        # ---------------------------------------------
        min_event = time_to_event.min()
        dt = 1.0 if min_event > 1.0 else min_event

        # ---------------------------------------------
        # RELIABILITY CALCULATION
        # ---------------------------------------------
        if dispatched_power < current_load - 1e-4:
            deficit = current_load - dispatched_power  # MW
            total_LOL_hours += dt
            total_LOEE += deficit * dt  # MWh

        total_system_cost += hourly_cost * dt

        # ---------------------------------------------
        # ADVANCE TIME
        # ---------------------------------------------
        current_time += dt
        time_to_event -= dt

        # ---------------------------------------------
        # HANDLE EVENTS
        # ---------------------------------------------
        events = np.where(time_to_event <= 1e-6)[0]

        for i in events:
            current_state[i] = 1 - current_state[i]

            if current_state[i] == 0:
                # Repaired → sample time to failure
                time_to_event[i] = -MTTF[i] * np.log(np.random.rand())
            else:
                # Failed → sample repair time
                time_to_event[i] = -MTTR[i] * np.log(np.random.rand())

        # Progress every 1000 years
        if int(current_time) % (HOURS_PER_YEAR * 100) < dt:
            print(f"Progress: Year {int(current_time // HOURS_PER_YEAR)}")

    # =====================================================
    # FINAL RESULTS
    # =====================================================
    LOLP = total_LOL_hours / TOTAL_HOURS
    LOLE = total_LOL_hours / NUM_YEARS
    LOEE = total_LOEE / NUM_YEARS
    avg_annual_cost = total_system_cost / NUM_YEARS

    print("\n========= FINAL RESULTS =========")
    print(f"LOLE (Hours/Year)        : {LOLE:.4f}")
    print(f"LOLP                     : {LOLP:.8f}")
    print(f"LOEE (MWh/Year)          : {LOEE:.2f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")
    print(f"Execution Time           : {datetime.now() - start_clock}")


# =========================================================
# 6. MAIN
# =========================================================
if __name__ == "__main__":

    Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup = load_data()
    run_smcs(Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup)

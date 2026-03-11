import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

# ==========================================
# 1. SIMULATION CONFIGURATION & CONSTANTS
# ==========================================
NUM_YEARS = 1000
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# Batch processing configuration to prevent Out-Of-Memory (OOM) errors 
# while maintaining the speed benefits of vectorization.
BATCH_YEARS = 50
HOURS_PER_BATCH = BATCH_YEARS * HOURS_PER_YEAR
NUM_BATCHES = NUM_YEARS // BATCH_YEARS

# Input file paths
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"
HYDRO_PROFILE_FILE = "../data/Monthly_Hydro_Profile.csv"


# ==========================================
# 2. DATA PREPARATION
# ==========================================
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads master data for generators and hourly load profile, then preprocesses them.
    Sorting by Merit Order is applied here so that cheapest generators are dispatched first.
    
    Returns:
        Tuple containing parameters for each generator and the load profile.
    """
    # Load generator data
    df_gen = pd.read_csv(GEN_DATA_FILE)

    # Sort generators by Merit Order (Cheapest unit cost first)
    cost_column = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_column).reset_index(drop=True)

    # Extract generator parameters as numpy arrays for fast computation
    unit_capacities = df_gen['Unit Capacity (MW)'].values.astype(float)
    mttf_hours = df_gen['MTTF (hours)'].values.astype(float)
    mttr_hours = df_gen['MTTR (hours)'].values.astype(float)
    unit_costs = df_gen[cost_column].values.astype(float)

    # Create boolean masks to easily identify Coal and Hydro plants later
    types_series = df_gen['TYPES'].astype(str).str.upper().str.strip()
    is_coal = np.array(types_series.str.contains('COAL'))
    is_hydro = np.array(types_series == 'HYDRO')

    # Load annual load curve data (8760 hours generally)
    df_load = pd.read_csv(LOAD_DATA_FILE)
    annual_load = df_load.iloc[:, 0].values.astype(float)

    # Load hydro hourly profile (24 hours x 12 months)
    df_hydro = pd.read_csv(HYDRO_PROFILE_FILE)
    # Extract the 24x12 matrix (assuming columns 1 to 12 are January to December)
    hydro_hourly_profile = df_hydro.iloc[:, 1:13].values.astype(float)

    # Create a mapping that tells us which month each hour of the year belongs to.
    # This assumes a standard non-leap year (8760 hours).
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for month_idx, days in enumerate(days_in_months):
        hours_in_month = days * 24
        month_lookup.extend([month_idx] * hours_in_month)
    month_lookup = np.array(month_lookup)

    # Map month and hour of day to get an 8760-hour array of hydro capacities
    hour_of_day_lookup = np.arange(8760) % 24
    hydro_capacity_base = hydro_hourly_profile[hour_of_day_lookup, month_lookup]

    return unit_capacities, mttf_hours, mttr_hours, is_hydro, is_coal, unit_costs, annual_load, hydro_capacity_base


# ==========================================
# 3. VECTORIZED SIMULATION ENGINE
# ==========================================
def run_vectorized_smcs(capacities: np.ndarray, mttf: np.ndarray, mttr: np.ndarray, 
                        is_hydro: np.ndarray, is_coal: np.ndarray, unit_costs: np.ndarray, 
                        annual_load: np.ndarray, hydro_capacity_base: np.ndarray):
    """
    Runs a vector-based Sequential Monte Carlo Simulation (SMCS) to evaluate power system reliability.
    """
    # ------------------------------------------------------------------
    # Step 1: Prepare Dispatch Order
    # ------------------------------------------------------------------
    # Force Base Load (Coal) to be dispatched first, followed by the rest 
    # (which are already sorted by cheapest cost in load_data).
    coal_indices = np.where(is_coal)[0]
    non_coal_indices = np.where(~is_coal)[0]
    dispatch_order = np.concatenate([coal_indices, non_coal_indices])

    # Reorder all generator arrays according to this physical dispatch sequence
    capacities = capacities[dispatch_order]
    mttf = mttf[dispatch_order]
    mttr = mttr[dispatch_order]
    is_hydro = is_hydro[dispatch_order]
    unit_costs = unit_costs[dispatch_order]
    num_generators = len(capacities)

    # Create batch-sized versions of load and hydro capacity to avoid re-generating them
    load_batch = np.tile(annual_load, BATCH_YEARS)
    hydro_capacity_batch = np.tile(hydro_capacity_base, BATCH_YEARS)

    print(f"Starting Vectorized SMCS for {NUM_YEARS} years...")
    start_time = datetime.now()

    # ------------------------------------------------------------------
    # Step 2: Generate All Failure/Repair Transitions Up-Front
    # ------------------------------------------------------------------
    # For every generator, we simulate its timeline of [Up -> Down -> Up -> Down...]
    # using exponential distributions based on MTTF and MTTR.
    transitions_list = []
    
    for i in range(num_generators):
        # Estimate how many Up/Down cycles we need for the entire simulation duration
        average_cycle_time = mttf[i] + mttr[i]
        estimated_cycles = int(TOTAL_HOURS / average_cycle_time * 1.5) + 100

        # Generate random variables for Time to Failure (TTF) and Time to Repair (TTR)
        # Formula: -Mean * ln(U) where U is uniform random [0, 1]
        time_to_fail = -mttf[i] * np.log(np.random.rand(estimated_cycles))
        time_to_repair = -mttr[i] * np.log(np.random.rand(estimated_cycles))

        # Interleave TTF and TTR into a single timeline array
        durations = np.empty(estimated_cycles * 2)
        durations[0::2] = time_to_fail    # Even indices are operation time (Up)
        durations[1::2] = time_to_repair  # Odd indices are repair time (Down)

        # Convert durations into continuous points in time where state transitions happen
        transition_times = np.cumsum(durations)

        # Safeguard: If our estimated cycles didn't cover the full timespan, generate more
        while transition_times[-1] < TOTAL_HOURS:
            extra_time_to_fail = -mttf[i] * np.log(np.random.rand(10))
            extra_time_to_repair = -mttr[i] * np.log(np.random.rand(10))
            
            extra_durations = np.empty(20)
            extra_durations[0::2] = extra_time_to_fail
            extra_durations[1::2] = extra_time_to_repair
            
            new_transition_times = transition_times[-1] + np.cumsum(extra_durations)
            transition_times = np.concatenate([transition_times, new_transition_times])

        transitions_list.append(transition_times)

    # ------------------------------------------------------------------
    # Step 3: Batch Processing to Calculate Reliability
    # ------------------------------------------------------------------
    # We track overall reliability metrics here
    total_loss_of_load_hours = 0.0      # LOLP related
    total_unserved_energy = 0.0         # LOEE related
    total_loss_of_load_events = 0       # LOLF related
    total_system_cost = 0.0             # Cost related
    was_failing_prev_batch = False      # Used to track frequency of events across batch boundaries

    for batch_idx in range(NUM_BATCHES):
        start_hour = batch_idx * HOURS_PER_BATCH
        end_hour = (batch_idx + 1) * HOURS_PER_BATCH
        hour_sequence = np.arange(start_hour, end_hour)

        # 3.1 Map continuous transition times to discrete hourly states
        # 0 represents DOWN (failed), 1 represents UP (working)
        hourly_states = np.zeros((num_generators, HOURS_PER_BATCH), dtype=np.int8)
        
        for i in range(num_generators):
            # `searchsorted` finds where the current hour falls in the transition timeline.
            # Even index means the generator is currently in TTF (Up), odd means TTR (Down).
            state_indices = np.searchsorted(transitions_list[i], hour_sequence)
            hourly_states[i, :] = (state_indices % 2 == 0).astype(np.int8)

        # 3.2 Calculate Base Availability (State * Capacity)
        unit_availability = hourly_states * capacities[:, None]

        # 3.3 Apply Hydro Power Constraints
        # Hydro generation is often limited by available water (hourly profile constraints)
        if np.any(is_hydro):
            hydro_avail = unit_availability[is_hydro, :]
            
            # Form cumulative sum of hydro generation sequentially
            cum_hydro = np.cumsum(hydro_avail, axis=0)

            # Cap the total cumulative hydro dispatch to the hourly profile limit
            cum_hydro_capped = np.minimum(cum_hydro, hydro_capacity_batch)

            # Reverse the cumulative sum to find what each individual unit is allowed to dispatch
            capped_hydro_avail = np.empty_like(hydro_avail)
            capped_hydro_avail[0, :] = cum_hydro_capped[0, :]
            if len(hydro_avail) > 1:
                capped_hydro_avail[1:, :] = cum_hydro_capped[1:, :] - cum_hydro_capped[:-1, :]

            # Overwrite original availability with the budget-capped availability
            unit_availability[is_hydro, :] = capped_hydro_avail

        # 3.4 Merit Order Dispatch (Satisfying the load)
        # Accumulate the available generation progressively through the dispatch order
        cum_availability = np.cumsum(unit_availability, axis=0)

        # Shift the array to know the capacity already provided *before* deciding on the current unit
        cum_availability_prev_unit = np.zeros_like(cum_availability)
        cum_availability_prev_unit[1:, :] = cum_availability[:-1, :]

        # Calculate remaining unmet load *before* this unit kicks in
        unmet_load_before_unit = np.maximum(0, load_batch - cum_availability_prev_unit)
        
        # Dispatch what is smaller: the unit's availability or the remaining load
        dispatched_power = np.minimum(unit_availability, unmet_load_before_unit)

        # 3.5 Evaluate Reliability Indices
        total_system_availability = cum_availability[-1, :]
        capacity_shortfall = np.maximum(0, load_batch - total_system_availability)
        
        # Consider it a failure event if shortfall is practically greater than 0
        is_failing_hour = capacity_shortfall > 1e-4

        total_loss_of_load_hours += np.sum(is_failing_hour)
        total_unserved_energy += np.sum(capacity_shortfall)

        # Track failure events (transitions from NOT failing to FAILING)
        is_failing_padded = np.insert(is_failing_hour, 0, was_failing_prev_batch)
        new_failure_events = is_failing_hour & ~is_failing_padded[:-1]
        
        total_loss_of_load_events += np.sum(new_failure_events)
        was_failing_prev_batch = is_failing_hour[-1]

        # 3.6 Evaluate Economic Impact (Cost)
        # dispatched_power is in MW, multiply by 1000 to get kW, then multiply by LKR/kWh
        hourly_costs = dispatched_power * unit_costs[:, None] * 1000
        total_system_cost += np.sum(hourly_costs)

        # Log Progress
        current_lolp = total_loss_of_load_hours / end_hour
        print(f"Completed: Batch {batch_idx + 1}/{NUM_BATCHES} ({BATCH_YEARS} years each), "
              f"Current LOLP: {current_lolp:.8f}")

    # ------------------------------------------------------------------
    # Step 4: Final Results Reporting
    # ------------------------------------------------------------------
    # Reliability Metrics Definition:
    # LOLP (Loss of Load Probability): Probability of load exceeding available generation
    # LOLE (Loss of Load Expectation): Expected hours per year load exceeds generation
    # LOEE (Loss of Energy Expectation): Expected unserved energy (MWh) per year
    # LOLF (Loss of Load Frequency): Expected number of failure events per year
    # LOLD (Loss of Load Duration): Average duration of a failure event

    lolp = total_loss_of_load_hours / TOTAL_HOURS
    lole = total_loss_of_load_hours / NUM_YEARS
    loee = total_unserved_energy / NUM_YEARS
    lolf = total_loss_of_load_events / NUM_YEARS
    lold = total_loss_of_load_hours / total_loss_of_load_events if total_loss_of_load_events > 0 else 0.0
    avg_annual_cost = total_system_cost / NUM_YEARS

    print("\n" + "="*30)
    print("        FINAL RESULTS         ")
    print("="*30)
    print(f"LOLE (Hours/Year)        : {lole:.4f}")
    print(f"LOLP                     : {lolp:.8f}")
    print(f"LOEE (MWh/Year)          : {loee:.2f}")
    print(f"LOLF (Events/Year)       : {lolf:.4f}")
    print(f"LOLD (Hours/Event)       : {lold:.4f}")
    print(f"Avg Annual System Cost   : LKR {avg_annual_cost:,.2f}")
    print(f"Execution Time           : {datetime.now() - start_time}")
    print("="*30)


if __name__ == "__main__":
    # Execute the simulation pipeline
    capacities, mttf, mttr, is_hydro, is_coal, costs, load, hydro_cap = load_data()
    run_vectorized_smcs(
        capacities=capacities, 
        mttf=mttf, 
        mttr=mttr, 
        is_hydro=is_hydro, 
        is_coal=is_coal, 
        unit_costs=costs, 
        annual_load=load, 
        hydro_capacity_base=hydro_cap
    )
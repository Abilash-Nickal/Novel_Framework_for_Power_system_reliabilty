import numpy as np
import math

# =========================================================
# 1. SYSTEM PARAMETERS & CONFIGURATION
# =========================================================
SIMULATION_YEARS = 500  # Number of Monte Carlo years to simulate
HOURS_PER_YEAR = 8760

# --- PHS Parameters (From Case Study C in the Paper) ---
PHS_CAPACITY_MW = 300.0  # PC: Generation and pumping capacity
PHS_STORAGE_MWH = 1000.0  # ES: Total energy storage capacity
PHS_EFFICIENCY = 0.70  # ETA: Pumping cycle efficiency (70%)
PHS_MTTF = 1960.0  # Mean Time To Failure for PHS (Hours)
PHS_MTTR = 40.0  # Mean Time To Repair for PHS (Hours)

# --- Conventional Generators Mock Data ---
# Format: (Capacity_MW, MTTF_hours, MTTR_hours)
CONV_GENERATORS = [
    (400, 1100, 150), (400, 1100, 150), (400, 1100, 150),  # Base load
    (350, 1500, 100), (350, 1500, 100), (350, 1500, 100),
    (150, 1960, 40), (150, 1960, 40), (150, 1960, 40),
    (50, 1000, 20), (50, 1000, 20), (50, 1000, 20)  # Peakers
]


# =========================================================
# 2. SYNTHETIC 8760-HOUR PROFILES (Load, Wind, Solar)
# =========================================================
def generate_8760_profiles():
    """
    Generates 8760-hour arrays mimicking the diurnal and seasonal
    variations discussed in the paper.
    """
    load = np.zeros(HOURS_PER_YEAR)
    solar = np.zeros(HOURS_PER_YEAR)
    wind = np.zeros(HOURS_PER_YEAR)

    for h in range(HOURS_PER_YEAR):
        day_of_year = h // 24
        hour_of_day = h % 24

        # 1. Load: High in daytime, dips at night. High in winter/summer.
        seasonal_load = 2200 + 400 * math.cos(2 * math.pi * (day_of_year - 15) / 365)  # Winter peak
        diurnal_load = 500 * math.sin(math.pi * (hour_of_day - 6) / 12) if 6 <= hour_of_day <= 18 else -200
        load[h] = seasonal_load + diurnal_load

        # 2. Solar: Zero at night, bell curve during day. Higher in summer.
        seasonal_solar = 1.0 - 0.3 * math.cos(2 * math.pi * day_of_year / 365)  # Summer peak
        if 6 <= hour_of_day <= 18:
            solar[h] = 400 * math.sin(math.pi * (hour_of_day - 6) / 12) * seasonal_solar

        # 3. Wind: Intermittent, generally higher in winter.
        seasonal_wind = 1.0 + 0.4 * math.cos(2 * math.pi * day_of_year / 365)
        wind[h] = max(0, 300 * seasonal_wind + np.random.normal(0, 50))

    return load, solar, wind


# =========================================================
# 3. MARKOV STATE GENERATOR (Eq. 1, 2, 3, 4)
# =========================================================
def generate_annual_states(capacity, mttf, mttr):
    """
    Simulates the stochastic Up/Down states of a generator for one full year.
    Returns an 8760-length array containing the available MW at each hour.
    """
    hourly_capacity = np.zeros(HOURS_PER_YEAR)
    current_time = 0.0
    state = 1  # 1 = UP, 0 = DOWN

    while current_time < HOURS_PER_YEAR:
        # Eq 3 & 4: Inverse transform method for exponential distribution
        rand_val = max(1e-9, min(0.999999, np.random.rand()))

        if state == 1:
            duration = -mttf * math.log(rand_val)
            avail_cap = capacity
        else:
            duration = -mttr * math.log(rand_val)
            avail_cap = 0.0

        start_idx = int(current_time)
        end_idx = min(HOURS_PER_YEAR, int(current_time + duration))

        if start_idx < HOURS_PER_YEAR:
            hourly_capacity[start_idx:end_idx] = avail_cap

        current_time += duration
        state = 1 - state  # Toggle state

    return hourly_capacity


# =========================================================
# 4. SEQUENTIAL MONTE CARLO SIMULATION ENGINE
# =========================================================
def run_smcs():
    print("Generating Chronological Load, Solar, and Wind Profiles...")
    load_8760, solar_8760, wind_8760 = generate_8760_profiles()

    total_loss_of_load_hours = 0
    total_unserved_energy = 0.0

    print(f"Starting SMCS for {SIMULATION_YEARS} years...")

    for year in range(SIMULATION_YEARS):
        # Step 2: Sequentially model conventional generators
        conv_8760 = np.zeros(HOURS_PER_YEAR)
        for cap, mttf, mttr in CONV_GENERATORS:
            conv_8760 += generate_annual_states(cap, mttf, mttr)

        # Also simulate the PHS unit's physical Markov states
        phs_state_8760 = generate_annual_states(1.0, PHS_MTTF, PHS_MTTR)  # 1.0 means UP

        # Initialize PHS Reservoir level (EP)
        # Let's assume it starts the year 50% full
        ep_mwh = PHS_STORAGE_MWH * 0.5

        loss_hours_this_year = 0

        # Step 3, 4, 5, 6, 7: Chronological Hourly Loop
        for h in range(HOURS_PER_YEAR):
            # Calculate total generation excluding PHS
            gen_total = conv_8760[h] + solar_8760[h] + wind_8760[h]
            current_load = load_8760[h]
            phs_is_up = (phs_state_8760[h] > 0)

            # Step 2 from paper (Eq. 5)
            surplus_power = gen_total - current_load
            deficit = 0.0

            # ---------------------------------------------------------
            # CHARGING THE PHS (Eq. 6, 7, 8, 9)
            # ---------------------------------------------------------
            if surplus_power > 0:
                if phs_is_up:
                    # We can only pump up to the PHS MW Capacity
                    pumping_power = min(surplus_power, PHS_CAPACITY_MW)

                    # We can only pump what fits in the reservoir (accounting for efficiency)
                    available_space_mwh = PHS_STORAGE_MWH - ep_mwh
                    max_pump_allowed = available_space_mwh / PHS_EFFICIENCY

                    pumping_power = min(pumping_power, max_pump_allowed)

                    # Update reservoir (Eq. 9)
                    ep_mwh += (pumping_power * PHS_EFFICIENCY)

            # ---------------------------------------------------------
            # DISCHARGING THE PHS (Eq. 10, 11, 12, 13, 14)
            # ---------------------------------------------------------
            elif surplus_power < 0:
                curtailment = -surplus_power  # How much power we are short

                if phs_is_up:
                    # We can only generate up to the PHS MW Capacity
                    generating_power = min(curtailment, PHS_CAPACITY_MW)

                    # We can only generate what water we have left
                    generating_power = min(generating_power, ep_mwh)

                    # Update reservoir (Eq. 14)
                    ep_mwh -= generating_power

                    # Calculate remaining deficit after PHS helps
                    deficit = curtailment - generating_power
                else:
                    # PHS is broken down, we suffer the full curtailment
                    deficit = curtailment

            # ---------------------------------------------------------
            # ADEQUACY EVALUATION (Step 7)
            # ---------------------------------------------------------
            if deficit > 1e-4:
                total_loss_of_load_hours += 1
                total_unserved_energy += deficit

        if (year + 1) % 50 == 0:
            print(f"  Finished Year {year + 1}/{SIMULATION_YEARS}...")

    # Step 8: Calculate final indices (Eq. 18, 19, 20)
    lole = total_loss_of_load_hours / SIMULATION_YEARS
    lolp = total_loss_of_load_hours / (SIMULATION_YEARS * HOURS_PER_YEAR)
    loee = total_unserved_energy / SIMULATION_YEARS

    print("\n" + "=" * 40)
    print(" GENERATION SYSTEM ADEQUACY INDICES")
    print("=" * 40)
    print(f" Simulation Years : {SIMULATION_YEARS}")
    print(f" PHS Capacity     : {PHS_CAPACITY_MW} MW")
    print(f" PHS Storage      : {PHS_STORAGE_MWH} MWh")
    print("-" * 40)
    print(f" LOLP : {lolp:.6f}")
    print(f" LOLE : {lole:.2f} Hours/Year")
    print(f" LOEE : {loee:.2f} MWh/Year")
    print("=" * 40)


if __name__ == "__main__":
    run_smcs()
import pandas as pd
import numpy as np

# --- FILE PATHS ---
GEN_FILE = "../data/CEB_GEN_MTTR_&_MTTF_for_each_unit.csv"
LOAD_FILE = "../data/SriLanka_Load_8760hr_repeat.csv"

# --- 1. Generator Data Conversion ---
try:
    df_gen = pd.read_csv(GEN_FILE)

    # Calculate FOR = MTTR / (MTTR + MTTF) because the JS simulation needs FOR and capacity.
    # The JS worker will calculate the final FOR, but we send all three components.
    Gen_df = df_gen[['Capacity (MW)', 'MTTF (hours)', 'MTTR (hours)']].astype(float)

    # Combine the three required columns into a list of lists: [[C1, MTTF1, MTTR1], [C2, MTTF2, MTTR2], ...]
    gen_data_list = Gen_df.values.tolist()

    # Convert to a JS-compatible string
    gen_js_array = str(gen_data_list).replace('[', '[\n    ').replace('],', '],\n    ').replace(']]', '\n]')

    print("--- 1. GENERATOR DATA ARRAY (Copy everything below) ---")
    print(f"const GENERATOR_DATA = {gen_js_array};\n")

except Exception as e:
    print(f"ERROR reading generator file: {e}")

# --- 2. Load Data Conversion ---
try:
    df_load = pd.read_csv(LOAD_FILE)

    # Get the load data (first column) and flatten it
    load_values = df_load.iloc[:, 0].astype(float).values

    # Ensure it is exactly 8760 hours (important for stability)
    HOURS_PER_YEAR = 8760
    if len(load_values) > HOURS_PER_YEAR:
        load_values = load_values[:HOURS_PER_YEAR]

    # Convert to a flat JavaScript array string
    load_js_array = str(load_values.tolist())

    print("--- 2. LOAD DATA ARRAY (Copy the full list of numbers) ---")
    print(f"const ANNUAL_LOAD_PROFILE = {load_js_array};\n")

except Exception as e:
    print(f"ERROR reading load file: {e}")

print("----------------------------------------------------------")
print("STEP 3: PASTE BOTH ARRAYS INTO 'reliability_dashboard.html'")
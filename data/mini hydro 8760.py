import pandas as pd
import numpy as np


def generate_hourly_mini_hydro():
    # 1. Data from the provided image (12 months of Mini Hydro generation in MW)
    monthly_values = [
        233.215427,  # January
        236.6899954,  # February
        276.2581406,  # March
        250.4085499,  # April
        279.4331773,  # May
        309.7457912,  # June
        289.9766951,  # July
        263.4381814,  # August
        256.6088573,  # September
        288.7186618,  # October
        323.4343925,  # November
        305.6122529  # December
    ]

    # 2. Define days in each month for a standard non-leap year (total 8760 hours)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    hourly_data = []

    # 3. Expand monthly values to hourly values
    for month_index, days in enumerate(month_days):
        hours_in_month = days * 24
        # Repeat the monthly average value for every hour in that month
        month_hours = [monthly_values[month_index]] * hours_in_month
        hourly_data.extend(month_hours)

    # 4. Verify total length is 8760
    if len(hourly_data) != 8760:
        print(f"Warning: Data length is {len(hourly_data)}, expected 8760.")
    else:
        print("Successfully generated 8760 hourly data points.")

    # 5. Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Hour': range(1, 8761),
        'Mini_Hydro_Generation_MW': hourly_data
    })

    output_filename = "MINI_HYDRO_GENERATION_8760.csv"
    df.to_csv(output_filename, index=False)
    print(f"File saved as: {output_filename}")


if __name__ == "__main__":
    generate_hourly_mini_hydro()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_lolp_comparison(csv_file='reliability_results.csv'):
    """
    Reads reliability results from a CSV file and plots the LOLP for different methods.

    Args:
        csv_file (str): The path to the CSV file containing the reliability results.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Convert Timestamp to datetime objects for potential time-based plotting
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    plt.figure(figsize=(12, 7))

    # Get unique methods from the 'Method' column
    methods = df['Method'].unique()

    # Define a list of markers and colors for better differentiation
    markers = ['o', 's', '^', 'd', 'p', 'h']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for i, method in enumerate(methods):
        method_df = df[df['Method'] == method].sort_values(by='Timestamp')
        # Use a cumulative count for the x-axis to represent experiment number,
        # as timestamps are not uniformly spaced and you requested "dynamical".
        # This gives a sequence of points for each method.
        x_values = range(1, len(method_df) + 1)
        plt.plot(x_values, method_df['LOLP'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-',
                 label=method)

    plt.xlabel('Experiment Number (Chronological for each Method)')
    plt.ylabel('LOLP (Loss of Load Probability)')
    plt.title('Comparison of LOLP Across Different Methods')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.yscale('log') # Use a logarithmic scale for LOLP as values can vary widely

    # Improve y-axis tick formatting for scientific notation if needed
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3)) # Adjust power limits as needed
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

# To run the function with your data:
plot_lolp_comparison('reliability_results.csv')
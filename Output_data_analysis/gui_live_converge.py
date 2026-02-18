import numpy as np # Numerical operations
import pandas as pd  # Data handling
from datetime import datetime # For timestamping
import matplotlib.pyplot as plt # Plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Embedding Matplotlib in Tkinter
import tkinter as tk # GUI framework
from tkinter import ttk, messagebox  # For styled widgets and dialogs
import threading # For running simulation in background
import queue # For thread-safe communication between simulation and GUI

# =========================================================
# 1. SIMULATION CONFIGURATION
# =========================-================================
NUM_YEARS = 10000 # Increase to 30000 for final results
HOURS_PER_YEAR = 8760
TOTAL_HOURS = NUM_YEARS * HOURS_PER_YEAR

# =========================================================
# 2. INPUT FILES
# =========================================================
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"

# =========================================================
# 3. MONTHLY HYDRO CAPACITY (MW)
# =========================================================
HYDRO_MONTHLY_CAP = np.array([853, 866, 1011, 916, 1023, 1133, 1061, 964, 939, 1057, 1184, 1118])


# =========================================================
# 4. DATA LOADING
# =========================================================
def load_data():
    df_gen = pd.read_csv(GEN_DATA_FILE)
    cost_col = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(float)
    MTTF = df_gen['MTTF (hours)'].values.astype(float)
    MTTR = df_gen['MTTR (hours)'].values.astype(float)
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_col].values.astype(float)

    df_load = pd.read_csv(LOAD_DATA_FILE)
    Annual_Load = df_load.iloc[:, 0].values.astype(float)

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_lookup = []
    for m, days in enumerate(month_days):
        month_lookup.extend([m] * (days * 24))
    month_lookup = np.array(month_lookup)

    return Gen, MTTF, MTTR, is_hydro, UnitCost, Annual_Load, month_lookup


# =========================================================
# 5. DYNAMIC GUI CLASS
# =========================================================
class ReliabilityGuiSMCS:
    def __init__(self, root, Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup):
        self.root = root
        self.root.title("SMCS Reliability Convergence Monitor")
        self.root.geometry("1100x800")
        self.root.configure(bg="#f8fafc")

        # Simulation Data
        self.Gen, self.MTTF, self.MTTR = Gen, MTTF, MTTR
        self.is_hydro, self.UnitCost, self.Load = is_hydro, UnitCost, Load
        self.month_lookup = month_lookup

        # Threading Queue
        self.update_queue = queue.Queue()

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=80)
        header.pack(fill="x")
        tk.Label(header, text="Sequential Monte Carlo Convergence Dashboard",
                 fg="white", bg="#003b5c", font=("Helvetica", 18, "bold")).pack(pady=20)

        # Control Panel
        control_frame = tk.Frame(self.root, bg="#f8fafc", padx=20, pady=10)
        control_frame.pack(fill="x")

        self.start_btn = tk.Button(control_frame, text="▶ Start Simulation", command=self.start_simulation,
                                   bg="#10b981", fg="white", font=("Helvetica", 12, "bold"),
                                   padx=20, pady=10, relief="flat", cursor="hand2")
        self.start_btn.pack(side="left")

        self.status_label = tk.StringVar(value="Status: Ready")
        tk.Label(control_frame, textvariable=self.status_label, bg="#f8fafc",
                 font=("Helvetica", 11), fg="#64748b").pack(side="left", padx=20)

        # Metrics Row
        metrics_frame = tk.Frame(self.root, bg="#f8fafc", padx=20)
        metrics_frame.pack(fill="x")

        self.year_var = tk.StringVar(value="Year: 0")
        self.lole_var = tk.StringVar(value="Current LOLE: -")
        self.cost_var = tk.StringVar(value="Avg Annual Cost: -")

        for var in [self.year_var, self.lole_var, self.cost_var]:
            lbl = tk.Label(metrics_frame, textvariable=var, font=("Courier", 12, "bold"),
                           bg="white", relief="groove", padx=15, pady=10, width=25)
            lbl.pack(side="left", padx=5, pady=10)

        # Plot Area
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.ax.set_title("Real-Time LOLE Convergence", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Study Duration (Years)")
        self.ax.set_ylabel("LOLE (Hours/Year)")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.line, = self.ax.plot([], [], color='#003b5c', linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

        # Periodic GUI Update Check
        self.root.after(100, self.process_queue)

    def start_simulation(self):
        self.start_btn.config(state="disabled", bg="#94a3b8")
        self.status_label.set("Status: Simulating...")

        sim_thread = threading.Thread(target=self.sim_worker, daemon=True)
        sim_thread.start()

    def sim_worker(self):
        # Local logic variables (keeping your logic exactly as provided)
        num_of_generators = len(self.Gen)
        current_state = np.zeros(num_of_generators, dtype=int)
        time_to_event = -self.MTTF * np.log(np.random.rand(num_of_generators))

        total_LOL_hours = 0.0
        total_LOEE = 0.0
        total_system_cost = 0.0
        current_time = 0.0
        last_reported_year = 0

        while current_time < TOTAL_HOURS:
            hour_of_year = int(current_time) % HOURS_PER_YEAR
            month_idx = self.month_lookup[hour_of_year]
            current_load = self.Load[hour_of_year]

            # Dispatch Logic (Original)
            up_mask = (current_state == 0)
            dispatched_power = 0.0
            hourly_cost = 0.0
            current_hydro_used = 0.0
            hydro_cap = HYDRO_MONTHLY_CAP[month_idx]

            for i in range(num_of_generators):
                if not up_mask[i]: continue
                if dispatched_power >= current_load: break
                needed = current_load - dispatched_power
                if self.is_hydro[i]:
                    potential = min(self.Gen[i], hydro_cap - current_hydro_used)
                    contribution = min(potential, needed)
                    if contribution > 0:
                        dispatched_power += contribution
                        current_hydro_used += contribution
                        hourly_cost += contribution * 1000 * self.UnitCost[i]
                else:
                    contribution = min(self.Gen[i], needed)
                    dispatched_power += contribution
                    hourly_cost += contribution * 1000 * self.UnitCost[i]

            min_event = time_to_event.min()
            dt = 1.0 if min_event > 1.0 else min_event

            if dispatched_power < current_load - 1e-4:
                deficit = current_load - dispatched_power
                total_LOL_hours += dt
                total_LOEE += deficit * dt

            total_system_cost += hourly_cost * dt
            current_time += dt
            time_to_event -= dt

            # Handle Events
            events = np.where(time_to_event <= 1e-6)[0]
            for i in events:
                current_state[i] = 1 - current_state[i]
                if current_state[i] == 0:
                    time_to_event[i] = -self.MTTF[i] * np.log(np.random.rand())
                else:
                    time_to_event[i] = -self.MTTR[i] * np.log(np.random.rand())

            # Update GUI data every 10 years
            current_year = int(current_time // HOURS_PER_YEAR)
            if current_year > last_reported_year and current_year % 10 == 0:
                lole = total_LOL_hours / current_year
                avg_cost = total_system_cost / current_year
                self.update_queue.put({
                    'year': current_year,
                    'lole': lole,
                    'cost': avg_cost,
                    'finished': False
                })
                last_reported_year = current_year

        # Final Completion Signal
        self.update_queue.put({
            'year': NUM_YEARS,
            'lole': total_LOL_hours / NUM_YEARS,
            'cost': total_system_cost / NUM_YEARS,
            'loee': total_LOEE / NUM_YEARS,
            'finished': True
        })

    def process_queue(self):
        try:
            while True:
                data = self.update_queue.get_nowait()

                # Update Text Labels
                self.year_var.set(f"Year: {data['year']:,}")
                self.lole_var.set(f"LOLE: {data['lole']:.4f}")
                self.cost_var.set(f"Avg Cost: {data['cost'] / 1e6:.2f}M LKR")

                # Update Graph
                xdata, ydata = self.line.get_data()
                new_x = np.append(xdata, data['year'])
                new_y = np.append(ydata, data['lole'])
                self.line.set_data(new_x, new_y)

                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()

                if data['finished']:
                    self.status_label.set("Status: Complete ✔")
                    messagebox.showinfo("Simulation Complete",
                                        f"Final LOLE: {data['lole']:.4f}\n"
                                        f"LOEE: {data['loee']:.2f} MWh/yr")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)


# =========================================================
# 6. MAIN
# =========================================================
if __name__ == "__main__":
    try:
        Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup = load_data()

        root = tk.Tk()
        app = ReliabilityGuiSMCS(root, Gen, MTTF, MTTR, is_hydro, UnitCost, Load, month_lookup)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
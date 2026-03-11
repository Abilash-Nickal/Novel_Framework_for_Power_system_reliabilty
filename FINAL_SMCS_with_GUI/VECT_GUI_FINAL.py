import tkinter as tk
from tkinter import ttk, messagebox
import multiprocessing
import concurrent.futures
import threading
import queue
import sys
import os
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd

# Matplotlib for GUI integration
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# ==========================================
# 1. WORKER FUNCTION (Vectorized + Live Updates)
# ==========================================
def worker_smcs(worker_id, years_to_run, batch_years, data_tuple, result_queue):
    """
    The vectorized worker function. Runs on a single CPU core.
    Sends batch updates back to the GUI via result_queue.
    """
    if years_to_run <= 0:
        if result_queue is not None:
            result_queue.put({"type": "done", "worker_id": worker_id})
        return

    HOURS_PER_YEAR = 8760

    # Unpack the data tuple
    Gen_orig, MTTF_orig, MTTR_orig, is_hydro_orig, is_coal_orig, UnitCost_orig, Load, Hydro_8760 = data_tuple

    # Re-seed the random number generator for this specific core
    np.random.seed(os.getpid() + worker_id + int(time.time() * 1000) % 100000)

    # --- PREPARE DISPATCH ORDER ---
    coal_idx = np.where(is_coal_orig)[0]
    non_coal_idx = np.where(~is_coal_orig)[0]
    dispatch_order = np.concatenate([coal_idx, non_coal_idx])

    Gen = Gen_orig[dispatch_order]
    MTTF = MTTF_orig[dispatch_order]
    MTTR = MTTR_orig[dispatch_order]
    is_hydro = is_hydro_orig[dispatch_order]
    UnitCost = UnitCost_orig[dispatch_order]
    num_gen = len(Gen)

    total_hours_to_run = years_to_run * HOURS_PER_YEAR

    # --- GENERATE TRANSITIONS UP-FRONT ---
    transitions_list = []
    for i in range(num_gen):
        num_cycles = int(total_hours_to_run / (MTTF[i] + MTTR[i]) * 1.5) + 100
        ttf = -MTTF[i] * np.log(np.random.rand(num_cycles))
        ttr = -MTTR[i] * np.log(np.random.rand(num_cycles))

        durations = np.empty(num_cycles * 2, dtype=np.float32)
        durations[0::2] = ttf
        durations[1::2] = ttr

        trans = np.cumsum(durations)

        while trans[-1] < total_hours_to_run:
            extra_ttf = -MTTF[i] * np.log(np.random.rand(10))
            extra_ttr = -MTTR[i] * np.log(np.random.rand(10))
            extra_durations = np.empty(20, dtype=np.float32)
            extra_durations[0::2] = extra_ttf
            extra_durations[1::2] = extra_ttr
            trans = np.concatenate([trans, trans[-1] + np.cumsum(extra_durations)])

        transitions_list.append(trans)

    was_failing_prev_batch = False

    # --- BATCH PROCESSING ---
    num_batches = int(np.ceil(years_to_run / batch_years))

    for b in range(num_batches):
        years_in_this_batch = min(batch_years, years_to_run - b * batch_years)
        hours_in_this_batch = years_in_this_batch * HOURS_PER_YEAR

        start_hour = b * batch_years * HOURS_PER_YEAR
        end_hour = start_hour + hours_in_this_batch
        hours = np.arange(start_hour, end_hour)

        Load_batch = np.tile(Load, years_in_this_batch)
        H_cap_batch = np.tile(Hydro_8760, years_in_this_batch)

        States = np.zeros((num_gen, hours_in_this_batch), dtype=np.int8)
        for i in range(num_gen):
            indices = np.searchsorted(transitions_list[i], hours)
            States[i, :] = (indices % 2 == 0).astype(np.int8)

        Avail = States * Gen[:, None]

        if np.any(is_hydro):
            hydro_avail = Avail[is_hydro, :]
            cum_hydro = np.cumsum(hydro_avail, axis=0)
            cum_hydro_capped = np.minimum(cum_hydro, H_cap_batch)

            capped_hydro_avail = np.empty_like(hydro_avail)
            capped_hydro_avail[0, :] = cum_hydro_capped[0, :]
            if len(hydro_avail) > 1:
                capped_hydro_avail[1:, :] = cum_hydro_capped[1:, :] - cum_hydro_capped[:-1, :]

            Avail[is_hydro, :] = capped_hydro_avail

        cum_Avail = np.cumsum(Avail, axis=0)
        cum_Avail_prev = np.zeros_like(cum_Avail)
        cum_Avail_prev[1:, :] = cum_Avail[:-1, :]

        Dispatched = np.minimum(Avail, np.maximum(0, Load_batch - cum_Avail_prev))

        Total_System_Avail = cum_Avail[-1, :]
        Shortfall = np.maximum(0, Load_batch - Total_System_Avail)
        is_failing = Shortfall > 1e-4

        batch_lol_hours = np.sum(is_failing)
        batch_loee = np.sum(Shortfall)

        is_failing_padded = np.insert(is_failing, 0, was_failing_prev_batch)
        new_events = is_failing & ~is_failing_padded[:-1]
        batch_lol_events = np.sum(new_events)
        was_failing_prev_batch = is_failing[-1]

        hourly_costs = Dispatched * UnitCost[:, None] * 1000
        batch_cost = np.sum(hourly_costs)

        # SEND DELTA TO GUI
        if result_queue is not None:
            result_queue.put({
                "type": "update",
                "worker_id": worker_id,
                "years": int(years_in_this_batch),
                "lol_hours": float(batch_lol_hours),
                "loee": float(batch_loee),
                "events": int(batch_lol_events),
                "cost": float(batch_cost)
            })

    if result_queue is not None:
        result_queue.put({"type": "done", "worker_id": worker_id})
    return True


# ==========================================
# 2. DATA LOADER
# ==========================================
def load_data(gen_file, load_file, hydro_file):
    df_gen = pd.read_csv(gen_file)
    cost_colm = 'Unit Cost (LKR/kWh)'
    df_gen = df_gen.sort_values(by=cost_colm).reset_index(drop=True)

    Gen = df_gen['Unit Capacity (MW)'].values.astype(np.float32)
    MTTF = df_gen['MTTF (hours)'].values.astype(np.float32)
    MTTR = df_gen['MTTR (hours)'].values.astype(np.float32)

    is_coal = (df_gen['TYPES'].str.upper().str.contains('COAL')).values
    is_hydro = (df_gen['TYPES'].str.upper().str.strip() == 'HYDRO').values
    UnitCost = df_gen[cost_colm].values.astype(np.float32)

    LOAD_COLUMN = "Modified_LC"
    df_load = pd.read_csv(load_file)
    Annual_Load = df_load[LOAD_COLUMN].values.astype(np.float32)

    try:
        df_hydro = pd.read_csv(hydro_file)
        if df_hydro.shape[1] > 12:
            df_hydro = df_hydro.iloc[:, -12:]
        hydro_matrix = df_hydro.values.astype(np.float32)
    except FileNotFoundError:
        print(f"[Warning] '{hydro_file}' not found. Using default flat 400MW profile.")
        hydro_matrix = np.full((24, 12), 400.0, dtype=np.float32)

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Hydro_8760 = []
    for m, days in enumerate(month_days):
        daily_profile = hydro_matrix[:, m]
        month_profile = np.tile(daily_profile, days)
        Hydro_8760.extend(month_profile)

    Hydro_8760 = np.array(Hydro_8760, dtype=np.float32)
    return Gen, MTTF, MTTR, is_hydro, is_coal, UnitCost, Annual_Load, Hydro_8760


# ==========================================
# 3. LIVE GRAPH GUI APPLICATION
# ==========================================
class AdvancedSMCSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("SMCS Reliability Monitor - Group 04")
        self.root.geometry("1300x850")
        self.root.configure(bg="#f1f5f9")

        # Multiprocessing Setup
        self.manager = multiprocessing.Manager()
        self.update_queue = self.manager.Queue()
        self.is_simulating = False

        # Tracking Aggregated Data (for the side panel)
        self.total_simulated_years = 0
        self.total_lol_hours = 0.0
        self.total_loee = 0.0
        self.total_cost = 0.0
        self.total_events = 0
        self.completed_workers = 0
        self.target_workers = 0
        self.start_time = 0

        # State for plotting Core subplots
        self.current_metric = "lole"
        self.core_data = {}  # Will hold individual core histories
        self.axes = []  # Matplotlib axes grid
        self.lines = {}  # Mapped lines by worker_id

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=60)
        header.pack(fill="x")
        tk.Label(header, text="VECTORIZED RELIABILITY MONITOR (SMCS)", fg="white",
                 bg="#003b5c", font=("Arial", 16, "bold")).pack(pady=15)

        # Config Panel (Top)
        config_frame = tk.Frame(self.root, bg="white", pady=10)
        config_frame.pack(fill="x", padx=20, pady=10)

        # Variables
        self.var_years = tk.IntVar(value=3000)
        self.var_batch = tk.IntVar(value=10)
        self.var_cores = tk.StringVar(value=str(multiprocessing.cpu_count()))
        self.var_gen_file = tk.StringVar(value="../data/CEB_GEN_Each_unit_Master_data.csv")
        self.var_load_file = tk.StringVar(value="../data/SRILANKAN_LOAD_CURVE_MODIFIED_2025.csv")
        self.var_hydro_file = tk.StringVar(value="../data/Monthly_Hydro_Profile.csv")

        # Row 1 Inputs
        tk.Label(config_frame, text="Total Years:", bg="white").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        tk.Entry(config_frame, textvariable=self.var_years, width=10).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(config_frame, text="Batch Years:", bg="white").grid(row=0, column=2, padx=5, pady=2, sticky="e")
        tk.Entry(config_frame, textvariable=self.var_batch, width=10).grid(row=0, column=3, padx=5, pady=2)

        tk.Label(config_frame, text="CPU Cores:", bg="white").grid(row=0, column=4, padx=5, pady=2, sticky="e")
        tk.Entry(config_frame, textvariable=self.var_cores, width=10).grid(row=0, column=5, padx=5, pady=2)

        # Row 2 Inputs
        tk.Label(config_frame, text="Gen Data:", bg="white").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        tk.Entry(config_frame, textvariable=self.var_gen_file, width=30).grid(row=1, column=1, columnspan=3, padx=5,
                                                                              pady=2, sticky="w")

        tk.Label(config_frame, text="Load Data:", bg="white").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        tk.Entry(config_frame, textvariable=self.var_load_file, width=30).grid(row=2, column=1, columnspan=3, padx=5,
                                                                               pady=2, sticky="w")

        # Action Buttons Area
        action_frame = tk.Frame(config_frame, bg="white")
        action_frame.grid(row=0, column=6, rowspan=3, padx=20)

        self.run_btn = tk.Button(action_frame, text="START SIMULATION", bg="#10b981", fg="white",
                                 font=("Arial", 11, "bold"), command=self.start_sim, padx=20, relief="flat",
                                 cursor="hand2")
        self.run_btn.pack(side="left", padx=10)

        self.status_lbl = tk.Label(action_frame, text="Status: Ready", bg="white", font=("Arial", 10), fg="#64748b")
        self.status_lbl.pack(side="left", padx=20)

        # Main Content Area (Side-by-Side)
        main_content = tk.Frame(self.root, bg="#f1f5f9")
        main_content.pack(fill="both", expand=True, padx=20, pady=5)

        # LEFT PANEL: Graphs
        left_panel = tk.Frame(main_content, bg="white", relief="flat", bd=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Placeholder canvas before simulation starts
        self.fig = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Ready to Start (Grid will populate dynamically)", fontsize=12, fontweight='bold')
        self.ax.set_ylabel("Hours/Year")
        self.ax.grid(True, alpha=0.3, linestyle='--')

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Graph Switcher Buttons
        switch_frame = tk.Frame(left_panel, bg="white", pady=10)
        switch_frame.pack(fill="x")
        tk.Label(switch_frame, text="Switch Graph View:", bg="white", font=("Arial", 9, "bold"), fg="#64748b").pack(
            side="left", padx=20)

        metrics_to_switch = [
            ("LOLE", "lole"),
            ("LOLP", "lolp"),
            ("LOEE", "loee"),
            ("Annual Cost", "cost")
        ]

        self.switch_btns = {}
        for label, key in metrics_to_switch:
            btn = tk.Button(switch_frame, text=label, command=lambda k=key: self.change_graph(k),
                            font=("Arial", 9), bg="#f1f5f9", relief="flat", padx=10)
            btn.pack(side="left", padx=5)
            self.switch_btns[key] = btn
        self.update_button_styles()

        # RIGHT PANEL: Numeric Dashboard Stack
        right_panel = tk.Frame(main_content, bg="#f1f5f9", width=300)
        right_panel.pack(side="right", fill="y")

        self.setup_numeric_stack(right_panel)

    def setup_numeric_stack(self, parent):
        scroll_container = tk.Frame(parent, bg="#f1f5f9")
        scroll_container.pack(fill="both", expand=True)

        self.metrics_info = {
            "y": ["Total System Years", "Years"],
            "lole": ["Total System LOLE", "Hrs/Yr"],
            "lolp": ["Total System LOLP", "Probability"],
            "loee": ["Total System LOEE", "MWh/Yr"],
            "events": ["Total System Events", "Count"],
            "cost": ["Avg Annual Cost", "LKR"],
            "sim_time": ["Simulation Time", "Seconds"]
        }
        self.vars = {}

        for key, info in self.metrics_info.items():
            card = tk.Frame(scroll_container, bg="white", relief="flat", bd=0, padx=15, pady=12)
            card.pack(fill="x", pady=(0, 10))

            tk.Label(card, text=info[0].upper(), bg="white", font=("Arial", 8, "bold"), fg="#94a3b8").pack(anchor="w")

            var = tk.StringVar(value="0.00" if key != "y" and key != "events" else "0")
            self.vars[key] = var
            tk.Label(card, textvariable=var, bg="white", font=("Courier New", 18, "bold"), fg="#0f172a").pack(
                anchor="w", pady=2)
            tk.Label(card, text=info[1], bg="white", font=("Arial", 8, "italic"), fg="#cbd5e1").pack(anchor="w")

    def change_graph(self, metric_key):
        self.current_metric = metric_key
        self.update_button_styles()

        titles = {"lole": "LOLE", "lolp": "LOLP", "loee": "LOEE", "cost": "Cost"}
        ylabels = {"lole": "Hours/Year", "lolp": "Probability", "loee": "MWh/Year", "cost": "LKR"}

        if not hasattr(self, 'axes') or len(self.axes) == 0:
            self.ax.set_title(f"{titles[metric_key]} Convergence", fontsize=12, fontweight='bold')
            self.ax.set_ylabel(ylabels[metric_key])
            self.canvas.draw()
            return

        # Update all subplots in the grid
        for i, ax in enumerate(self.axes):
            ax.set_title(f"Core {i + 1}: {titles[metric_key]}", fontsize=9, fontweight='bold')
            ax.set_ylabel(ylabels[metric_key], fontsize=8)

            if i in self.core_data:
                history = self.core_data[i]["history"]
                if history["y"]:
                    self.lines[i].set_data(history["y"], history[metric_key])
                    ax.relim()
                    ax.autoscale_view()

        self.canvas.draw()

    def update_button_styles(self):
        for key, btn in self.switch_btns.items():
            if key == self.current_metric:
                btn.config(bg="#003b5c", fg="white")
            else:
                btn.config(bg="#f1f5f9", fg="#0f172a")

    def start_sim(self):
        # Validate inputs
        try:
            total_years = self.var_years.get()
            batch_years = self.var_batch.get()
            core_input = self.var_cores.get().strip()

            gen_file = self.var_gen_file.get()
            load_file = self.var_load_file.get()
            hydro_file = self.var_hydro_file.get()

            print("Loading Data...")
            data_tuple = load_data(gen_file, load_file, hydro_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data or read settings:\n{str(e)}")
            return

        self.run_btn.config(state="disabled", bg="#94a3b8")
        self.status_lbl.config(text="Status: Simulating...")

        # Setup Parallel Jobs
        available_cores = multiprocessing.cpu_count()
        if core_input.isdigit() and int(core_input) > 0:
            num_cores = min(int(core_input), available_cores)
        else:
            num_cores = available_cores

        self.target_workers = num_cores

        # Initialize the Plot Grid for all cores
        self.fig.clear()
        cols = math.ceil(math.sqrt(num_cores))
        rows = math.ceil(num_cores / cols)

        self.axes = []
        self.lines = {}

        titles = {"lole": "LOLE", "lolp": "LOLP", "loee": "LOEE", "cost": "Cost"}
        ylabels = {"lole": "Hours/Year", "lolp": "Probability", "loee": "MWh/Year", "cost": "LKR"}
        m_title = titles[self.current_metric]
        m_ylab = ylabels[self.current_metric]

        for i in range(num_cores):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            ax.set_title(f"Core {i + 1}: {m_title}", fontsize=9, fontweight='bold')
            ax.set_ylabel(m_ylab, fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)

            line, = ax.plot([], [], color="#003b5c", lw=1.5)
            ax.grid(True, alpha=0.3, linestyle='--')

            self.axes.append(ax)
            self.lines[i] = line

        self.fig.tight_layout()
        self.canvas.draw()

        # Reset tracking data dynamically per core
        self.core_data = {}
        for i in range(num_cores):
            self.core_data[i] = {
                "years": 0, "lol_hours": 0.0, "loee": 0.0, "cost": 0.0, "events": 0,
                "history": {"y": [], "lole": [], "lolp": [], "loee": [], "cost": []}
            }

        # Reset Total Metrics Trackers
        self.total_simulated_years = 0
        self.total_lol_hours = 0.0
        self.total_loee = 0.0
        self.total_cost = 0.0
        self.total_events = 0
        self.completed_workers = 0

        years_per_core = total_years // num_cores
        remainder_years = total_years % num_cores

        jobs = []
        for i in range(num_cores):
            years_for_this_worker = years_per_core + (remainder_years if i == 0 else 0)
            jobs.append((i, years_for_this_worker, batch_years, data_tuple))

        self.start_time = time.time()
        self.is_simulating = True

        # Start Worker Pool in background thread
        threading.Thread(target=self.launch_workers, args=(jobs, num_cores), daemon=True).start()

        # Start checking queue on UI thread
        self.root.after(100, self.check_queue)

    def launch_workers(self, jobs, num_cores):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(worker_smcs, job[0], job[1], job[2], job[3], self.update_queue) for job in jobs]
            concurrent.futures.wait(futures)

    def check_queue(self):
        if not self.is_simulating:
            return

        graph_needs_update = False
        try:
            # Drain up to 100 items from the queue per UI tick to prevent freezing
            for _ in range(100):
                msg = self.update_queue.get_nowait()

                if msg["type"] == "update":
                    wid = msg["worker_id"]

                    # 1. Update Core-Specific Data
                    c_data = self.core_data[wid]
                    c_data["years"] += msg["years"]
                    c_data["lol_hours"] += msg["lol_hours"]
                    c_data["loee"] += msg["loee"]
                    c_data["cost"] += msg["cost"]
                    c_data["events"] += msg["events"]

                    # Calculate individual indices for this core
                    cy = c_data["years"]
                    c_lole = c_data["lol_hours"] / cy if cy > 0 else 0.0
                    c_lolp = c_data["lol_hours"] / (cy * 8760) if cy > 0 else 0.0
                    c_loee = c_data["loee"] / cy if cy > 0 else 0.0
                    c_cost = c_data["cost"] / cy if cy > 0 else 0.0

                    c_data["history"]["y"].append(cy)
                    c_data["history"]["lole"].append(c_lole)
                    c_data["history"]["lolp"].append(c_lolp)
                    c_data["history"]["loee"].append(c_loee)
                    c_data["history"]["cost"].append(c_cost)

                    # 2. Update Global System Tracking
                    self.total_simulated_years += msg["years"]
                    self.total_lol_hours += msg["lol_hours"]
                    self.total_loee += msg["loee"]
                    self.total_cost += msg["cost"]
                    self.total_events += msg["events"]

                    ty = self.total_simulated_years
                    if ty > 0:
                        t_lole = self.total_lol_hours / ty
                        t_lolp = self.total_lol_hours / (ty * 8760)
                        t_loee = self.total_loee / ty
                        t_cost = self.total_cost / ty
                    else:
                        t_lole = t_lolp = t_loee = t_cost = 0.0

                    self.vars["y"].set(f"{ty:,}")
                    self.vars["lole"].set(f"{t_lole:.4f}")
                    self.vars["lolp"].set(f"{t_lolp:.6f}")
                    self.vars["loee"].set(f"{t_loee:.2f}")
                    self.vars["events"].set(f"{self.total_events:,}")
                    self.vars["cost"].set(f"{t_cost / 1e6:.2f}M")

                    elapsed = time.time() - self.start_time
                    self.vars["sim_time"].set(f"{elapsed:.1f}s")

                    graph_needs_update = True

                elif msg["type"] == "done":
                    self.completed_workers += 1
                    if self.completed_workers >= self.target_workers:
                        self.is_simulating = False
                        self.run_btn.config(state="normal", bg="#10b981")
                        self.status_lbl.config(text="Status: Complete ✔")
                        messagebox.showinfo("Done", "Reliability Study Completed.")
                        graph_needs_update = True

        except queue.Empty:
            pass

        # Batch UI Graph Draw
        if graph_needs_update:
            for i, ax in enumerate(self.axes):
                if i in self.core_data:
                    hist = self.core_data[i]["history"]
                    if hist["y"]:
                        self.lines[i].set_data(hist["y"], hist[self.current_metric])
                        ax.relim()
                        ax.autoscale_view()
            self.canvas.draw()

        if self.is_simulating:
            self.root.after(100, self.check_queue)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = AdvancedSMCSGui(root)
    root.mainloop()
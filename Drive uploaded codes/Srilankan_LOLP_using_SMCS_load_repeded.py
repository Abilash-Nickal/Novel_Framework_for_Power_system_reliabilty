import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- File Paths (no input prompt) ---
GEN_DATA_FILE = "../data/CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"
HOURS_PER_YEAR = 8760


# --- Data Loading Function ---
def load_data():
    """Loads generator and load data from CSVs."""
    try:
        df_gen = pd.read_csv(GEN_DATA_FILE, header=0)
        Gen_df = df_gen[['Unit Capacity (MW)', 'MTTR (hours)', 'MTTF (hours)']].astype(float)
        Gen = Gen_df['Unit Capacity (MW)'].values
        MTTF = Gen_df['MTTF (hours)'].values
        MTTR = Gen_df['MTTR (hours)'].values

        df_load = pd.read_csv(LOAD_DATA_FILE, header=0)
        Annual_Load_Profile = df_load.iloc[:, 0].astype(float).values

        # Repeat or trim to 8760 hours
        if len(Annual_Load_Profile) < HOURS_PER_YEAR:
            repeats = int(np.ceil(HOURS_PER_YEAR / len(Annual_Load_Profile)))
            Annual_Load_Profile = np.tile(Annual_Load_Profile, repeats)[:HOURS_PER_YEAR]
        else:
            Annual_Load_Profile = Annual_Load_Profile[:HOURS_PER_YEAR]

        return Gen, MTTF, MTTR, Annual_Load_Profile
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


# --- Sequential Monte Carlo Simulation Thread ---
def run_smcs_thread(gui, num_years):
    Gen, MTTF, MTTR, Annual_Load_Profile = load_data()
    if Gen is None:
        gui.update_status("❌ Error loading CSV data. Simulation aborted.")
        return

    total_study_hours = num_years * HOURS_PER_YEAR
    state = np.zeros(len(Gen), dtype=int)
    time_to_next_event = -MTTF * np.log(np.random.rand(len(Gen)))
    total_LOL_hours = 0
    current_time = 0

    start_time = datetime.now()
    gui.update_status("Simulation started...")

    while current_time < total_study_hours:
        availableGen = np.sum(Gen[state == 0])
        annual_hour = int(current_time) % HOURS_PER_YEAR
        currentLoad = Annual_Load_Profile[annual_hour]

        min_time_step = np.min(time_to_next_event)
        time_step = min(1.0, min_time_step)

        if availableGen < currentLoad:
            total_LOL_hours += time_step

        current_time += time_step
        time_to_next_event -= time_step

        # Units with event completed
        changing_units = np.where(time_to_next_event <= 1e-6)[0]
        for i in changing_units:
            state[i] = 1 - state[i]
            time_to_next_event[i] = -MTTF[i] * np.log(np.random.rand()) if state[i] == 0 else -MTTR[i] * np.log(np.random.rand())

        # Update progress in GUI every 1000 hours
        if current_time % (HOURS_PER_YEAR * 1000) <= time_step and current_time > 0:
            gui.update_progress(current_time, total_study_hours, total_LOL_hours / current_time)

    # Results
    LOLP = total_LOL_hours / total_study_hours
    LOLE = total_LOL_hours / num_years
    duration = datetime.now() - start_time
    gui.show_results(total_LOL_hours, LOLP, LOLE, duration)


# --- Tkinter GUI ---
class SMCS_GUI:
    def __init__(self, root):
        self.root = root
        root.title("Sequential Monte Carlo Simulator")
        root.geometry("750x500")
        root.configure(bg="#f0f4f8")

        # Title Label
        tk.Label(
            root,
            text="Sequential Monte Carlo Simulator\nPower System Reliability",
            font=("Poppins", 18, "bold"),
            bg="#f0f4f8"
        ).pack(pady=10)

        # Input: NUM_YEARS
        frame = tk.Frame(root, bg="#f0f4f8")
        frame.pack(pady=5)
        tk.Label(frame, text="Enter Number of Years:", font=("Poppins", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
        self.year_input = tk.Entry(frame, width=20, font=("Poppins", 12))
        self.year_input.insert(0, "10000")
        self.year_input.pack(side=tk.LEFT)

        # Progress Bar
        self.progress = ttk.Progressbar(root, length=650, mode='determinate')
        self.progress.pack(pady=15)

        # Status
        self.status_text = tk.StringVar()
        self.status_text.set("Ready.")
        tk.Label(root, textvariable=self.status_text, font=("Poppins", 12), bg="#f0f4f8").pack(pady=5)

        # Live LOLP
        self.lolp_label = tk.StringVar()
        self.lolp_label.set("Live LOLP: -")
        tk.Label(root, textvariable=self.lolp_label, font=("Poppins", 12), bg="#f0f4f8").pack()

        # Start Button
        ttk.Button(root, text="▶ Run Simulation", width=25, command=self.start_simulation).pack(pady=15)

        # Results
        tk.Label(root, text="Final Results", font=("Poppins", 15, "bold"), bg="#f0f4f8").pack(pady=10)
        self.result_box = tk.Text(root, height=7, width=80, font=("Consolas", 11))
        self.result_box.pack()

    # GUI update functions
    def update_progress(self, current, total, lolp):
        pct = (current / total) * 100
        self.progress['value'] = pct
        self.lolp_label.set(f"Live LOLP: {lolp:.8f}")
        self.status_text.set(f"Running... {pct:.2f}% Completed")
        self.root.update_idletasks()

    def update_status(self, msg):
        self.status_text.set(msg)
        self.root.update_idletasks()

    def show_results(self, total_LOL_hours, LOLP, LOLE, duration):
        self.progress['value'] = 100
        self.status_text.set("Simulation Completed ✔")
        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, f"Total LOL Hours: {total_LOL_hours:,.2f}\n")
        self.result_box.insert(tk.END, f"LOLP: {LOLP:.8f}\n")
        self.result_box.insert(tk.END, f"LOLE (hours/year): {LOLE:.2f}\n")
        self.result_box.insert(tk.END, f"Simulation Duration: {duration}\n")
        messagebox.showinfo("Simulation Completed", "Sequential Monte Carlo Simulation Finished Successfully!")

    def start_simulation(self):
        try:
            num_years = int(self.year_input.get())
            if num_years <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a positive integer for number of years.")
            return

        self.progress['value'] = 0
        self.result_box.delete(1.0, tk.END)
        self.status_text.set("Starting simulation...")

        t = threading.Thread(target=run_smcs_thread, args=(self, num_years))
        t.start()


# --- Run GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = SMCS_GUI(root)
    root.mainloop()

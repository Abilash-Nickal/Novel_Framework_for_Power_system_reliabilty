import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
import tkinter.ttk as ttk



# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
GEN_DATA_FILE = "CEB_GEN_Each_unit_Master_data.csv"
LOAD_DATA_FILE = "SRILANKAN_LOAD_CURVE_MODIFIED.csv"

HOURS_PER_YEAR = 8760


# -------------------------------------------------------
# LOAD FUNCTIONS
# -------------------------------------------------------

def load_generator_data(filepath):
    df = pd.read_csv(filepath, usecols=['Unit Capacity (MW)', 'Unit FOR'])
    Gen = df['Unit Capacity (MW)'].astype(float).values
    FOR = df['Unit FOR'].astype(float).clip(0.0, 1.0).values
    return Gen, FOR


def load_annual_load_profile(filepath):
    df = pd.read_csv(filepath)
    return df.iloc[:, 0].astype(float).values


# -------------------------------------------------------
# SIMULATION THREAD FUNCTION
# -------------------------------------------------------

def run_monte_carlo_thread(gui, iterations):
    try:
        Gen, FOR = load_generator_data(GEN_DATA_FILE)
        Load = load_annual_load_profile(LOAD_DATA_FILE)
    except Exception as e:
        gui.update_status(f"❌ Error loading files:\n{e}")
        return

    gui.update_status("Files loaded. Running simulation...")

    Gen = np.array(Gen)
    FOR = np.array(FOR)

    H = 0
    N = 0
    start_time = datetime.now()

    for i in range(iterations):
        N += 1

        outage_mask = np.random.random(len(Gen)) > FOR
        availableGen = np.sum(Gen * outage_mask)

        currentLoad = Load[np.random.randint(0, len(Load))]

        if currentLoad > availableGen:
            H += 1

        # GUI update every 1M iterations
        if i % 1_000_000 == 0 and i > 0:
            gui.update_progress(i, iterations, H / N)

    duration = datetime.now() - start_time
    LOLP = H / N
    LOLE = LOLP * HOURS_PER_YEAR

    gui.show_results(LOLP, LOLE, duration)


# -------------------------------------------------------
# GUI CLASS
# -------------------------------------------------------

class ReliabilityGUI:

    def __init__(self, root):
        self.root = root
        root.title(" Monte Carlo Reliability Simulator-NSCMS")
        root.geometry("700x500")
        root.resizable(True, True)

        tk.Label(
            root,
            text="Power System Reliability Simulator Using\n Non-Sequential Monte Carlo",
            font=("Poppins", 20, "bold")
        ).pack(pady=10)

        # -----------------------------
        # INPUT: NUM ITERATIONS
        # -----------------------------
        frame = tk.Frame(root)
        frame.pack(pady=5)

        tk.Label(frame, text="Enter Number of Iterations:",
                 font=("Poppins", 12)).pack(side=tk.LEFT, padx=5)

        self.iter_input = tk.Entry(frame, width=20, font=("Poppins", 12))
        self.iter_input.insert(0, "1000000")  # default
        self.iter_input.pack(side=tk.LEFT)

        # -----------------------------
        # PROGRESS BAR
        # -----------------------------
        self.progress = ttk.Progressbar(root, length=600, mode='determinate')
        self.progress.pack(pady=15)

        # Status text
        self.status_text = tk.StringVar()
        self.status_text.set("Ready.")
        tk.Label(root, textvariable=self.status_text,
                 font=("Poppins", 12)).pack(pady=10)

        # Live LOLP
        self.lolp_label = tk.StringVar()
        self.lolp_label.set("Live LOLP: -")
        tk.Label(root, textvariable=self.lolp_label,
                 font=("Poppins", 12,)).pack()

        # Start button
        ttk.Button(root, text="▶ Run Simulation", width=25,
                   command=self.start_simulation).pack(pady=15)

        # Results
        tk.Label(root, text="Final Results",
                 font=("Poppins", 15, "bold")).pack(pady=10)

        self.result_box = tk.Text(root, height=7, width=80, font=("Consolas", 11))
        self.result_box.pack()

    # ----------------------------------------------------------------
    # GUI UPDATE FUNCTIONS
    # ----------------------------------------------------------------

    def update_progress(self, current, total, lolp):
        pct = (current / total) * 100
        self.progress['value'] = pct
        self.lolp_label.set(f"Live LOLP: {lolp:.8f}")
        self.status_text.set(f"Running... {pct:.2f}% Completed")
        self.root.update_idletasks()

    def update_status(self, msg):
        self.status_text.set(msg)
        self.root.update_idletasks()

    def show_results(self, lolp, lole, duration):
        self.progress['value'] = 100
        self.status_text.set("Simulation Completed ✔")

        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, f"Final LOLP (Probability): {lolp:.10f}\n")
        self.result_box.insert(tk.END, f"Final LOLE (Expected Hours/Year): {lole:.2f} hours/year\n")
        self.result_box.insert(tk.END, f"Simulation Duration: {duration}\n")

        messagebox.showinfo("Done", "Simulation completed successfully!")

    # ----------------------------------------------------------------
    # START BUTTON HANDLER
    # ----------------------------------------------------------------

    def start_simulation(self):
        try:
            iterations = int(self.iter_input.get())
            if iterations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for iterations.")
            return

        self.progress['value'] = 0
        self.status_text.set("Starting simulation...")
        self.result_box.delete(1.0, tk.END)

        t = threading.Thread(target=run_monte_carlo_thread,
                             args=(self, iterations))
        t.start()


# -------------------------------------------------------
# RUN GUI
# -------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    gui = ReliabilityGUI(root)
    root.mainloop()

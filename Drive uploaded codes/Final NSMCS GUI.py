import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
GEN_DATA_FILE = "CEB_GEN_Each_unit_Master_data.csv"

# Defining two different load datasets
LOAD_FILE_A = "SRILANKAN_LOAD_CURVE_MODIFIED.csv"
LOAD_FILE_B = "SriLanka_Load_8760hr_Random.csv"

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "NSMCS_Reliability_result.csv")

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

def run_monte_carlo_thread(gui, iterations, selected_load_path):
    try:
        Gen, FOR = load_generator_data(GEN_DATA_FILE)
        # Now using the path passed from the GUI selection
        Load = load_annual_load_profile(selected_load_path)
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
        if gui.stop_event.is_set():
            gui.update_status("Simulation stopped by user.")
            break

        while gui.pause_event.is_set() and not gui.stop_event.is_set():
            gui.update_status("Simulation paused.")
            gui.root.after(200)

        if gui.stop_event.is_set():
            gui.update_status("Simulation stopped by user.")
            break

        N += 1

        outage_mask = np.random.random(len(Gen)) > FOR
        availableGen = np.sum(Gen * outage_mask)

        # Randomly sample from the selected load profile
        currentLoad = Load[np.random.randint(0, len(Load))]

        if currentLoad > availableGen:
            H += 1

        # GUI update every 1M iterations to keep it responsive
        if i % 1_000_000 == 0 and i > 0:
            gui.update_progress(i, iterations, H / N)

    duration = datetime.now() - start_time
    if N > 0 and not gui.stop_event.is_set():
        LOLP = H / N
        LOLE = LOLP * HOURS_PER_YEAR
        gui.show_results(LOLP, LOLE, duration, iterations, selected_load_path)
    else:
        gui.run_btn.config(state="normal")
        gui.pause_btn.config(state="disabled")
        gui.stop_btn.config(state="disabled")


# -------------------------------------------------------
# GUI CLASS
# -------------------------------------------------------

class ReliabilityGUI:
    def __init__(self, root):
        self.root = root
        root.title("Monte Carlo CEB Reliability Simulator - NSCMS")
        root.geometry("750x650")

        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.sim_thread = None

        # Main Title
        tk.Label(
            root,
            text="Power System Reliability Simulator",
            font=("Poppins", 20, "bold"),
            fg="#005088"
        ).pack(pady=(20, 5))

        tk.Label(
            root,
            text="Non-Sequential Monte Carlo (NSMCS) Method",
            font=("Poppins", 12, "italic")
        ).pack(pady=(0, 20))

        # -----------------------------
        # INPUT: NUM ITERATIONS
        # -----------------------------
        input_frame = tk.LabelFrame(root, text=" Simulation Settings ", font=("Poppins", 10, "bold"), padx=20, pady=10)
        input_frame.pack(pady=10, fill="x", padx=40)

        tk.Label(input_frame, text="Number of Iterations:", font=("Poppins", 11)).grid(row=0, column=0, sticky="w")
        self.iter_input = tk.Entry(input_frame, width=20, font=("Consolas", 11))
        self.iter_input.insert(0, "1000000")
        self.iter_input.grid(row=0, column=1, padx=10)

        # -----------------------------
        # DATASET SELECTION (The Update)
        # -----------------------------
        tk.Label(input_frame, text="Select Load Dataset:", font=("Poppins", 11)).grid(row=1, column=0, sticky="w",
                                                                                      pady=(15, 0))

        self.load_choice = tk.StringVar(value=LOAD_FILE_A)

        rb1 = tk.Radiobutton(input_frame, text="Base Load Profile (Modified) only thermal & hydro", variable=self.load_choice,
                             value=LOAD_FILE_A, font=("Poppins", 10))
        rb1.grid(row=1, column=1, sticky="w", pady=(15, 0))

        rb2 = tk.Radiobutton(input_frame, text="Actual Load Profile Both Conv & NonConv", variable=self.load_choice,
                             value=LOAD_FILE_B, font=("Poppins", 10))
        rb2.grid(row=2, column=1, sticky="w")

        # -----------------------------
        # SAVE OPTIONS
        # -----------------------------
        self.save_to_csv = tk.BooleanVar(value=True)
        self.csv_path_var = tk.StringVar(value=RESULTS_CSV)

        save_check = tk.Checkbutton(
            input_frame,
            text="Save results to CSV",
            variable=self.save_to_csv,
            font=("Poppins", 10)
        )
        save_check.grid(row=3, column=0, sticky="w", pady=(15, 0))

        csv_entry = tk.Entry(input_frame, textvariable=self.csv_path_var, width=40, font=("Consolas", 10))
        csv_entry.grid(row=3, column=1, sticky="w", pady=(15, 0))

        # -----------------------------
        # PROGRESS & STATUS
        # -----------------------------
        self.progress = ttk.Progressbar(root, length=600, mode='determinate')
        self.progress.pack(pady=20)

        self.status_text = tk.StringVar(value="System Ready.")
        tk.Label(root, textvariable=self.status_text, font=("Poppins", 11, "bold"), fg="#334155").pack()

        self.lolp_label = tk.StringVar(value="Live LOLP: -")
        tk.Label(root, textvariable=self.lolp_label, font=("Consolas", 12), fg="#b91c1c").pack(pady=5)

        # Start button
        style = ttk.Style()
        style.configure("TButton", font=("Poppins", 12, "bold"))
        self.run_btn = ttk.Button(root, text="▶ Run Simulation", width=22, command=self.start_simulation)
        self.run_btn.pack(pady=8)

        # Pause/Stop Controls
        control_frame = tk.Frame(root)
        control_frame.pack(pady=(0, 12))

        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", width=12, command=self.toggle_pause, state="disabled")
        self.pause_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = ttk.Button(control_frame, text="■ Stop", width=12, command=self.stop_simulation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        # Results Area
        tk.Label(root, text="Final Output Report", font=("Poppins", 13, "bold")).pack()
        self.result_box = tk.Text(root, height=8, width=85, font=("Consolas", 10), bg="#f8fafc", padx=10, pady=10)
        self.result_box.pack(pady=10)

    def update_progress(self, current, total, lolp):
        pct = (current / total) * 100
        self.progress['value'] = pct
        self.lolp_label.set(f"Live LOLP : {lolp:.12f}")
        self.status_text.set(f"Processing... {pct:.2f}% Complete")
        self.root.update_idletasks()

    def update_status(self, msg):
        self.status_text.set(msg)
        self.root.update_idletasks()

    def show_results(self, lolp, lole, duration, iterations, selected_load_path):
        self.progress['value'] = 100
        self.status_text.set("Simulation Completed ✔")
        self.run_btn.config(state="normal")  # Re-enable button

        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, f"{'=' * 60}\n")
        self.result_box.insert(tk.END, f" LOAD PROFILE: {os.path.basename(self.load_choice.get())}\n")
        self.result_box.insert(tk.END, f"{'=' * 60}\n")
        self.result_box.insert(tk.END, f" Final LOLP (Probability):        {lolp:.12f}\n")
        self.result_box.insert(tk.END, f" Final LOLE (Hours/Year):         {lole:.2f}\n")
        duration_str = str(duration).split(".")[0]
        self.result_box.insert(tk.END, f" Computation Time:                {duration_str}\n")
        self.result_box.insert(tk.END, f"{'=' * 60}\n")

        if self.save_to_csv.get():
            try:
                self.save_results_csv(iterations, selected_load_path, lolp, lole, duration)
            except Exception as e:
                messagebox.showerror("CSV Save Error", f"Failed to save CSV:\n{e}")
                return

        messagebox.showinfo("Simulation Completed", "Non-Sequential Monte Carlo Simulation Finished.")

    def save_results_csv(self, iterations, selected_load_path, lolp, lole, duration):
        file_path = self.csv_path_var.get().strip() or RESULTS_CSV
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        load_type = "A" if selected_load_path == LOAD_FILE_A else "B" if selected_load_path == LOAD_FILE_B else os.path.basename(selected_load_path)
        row = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": int(iterations),
            "method": "NSMCS",
            "load data type": load_type,
            "LOLP": f"{lolp:.8f}",
            "LOLE": f"{lole:.8f}",
            "time taken": str(duration).split(".")[0],
        }

        df = pd.DataFrame([row])
        write_header = not os.path.exists(file_path)
        df.to_csv(file_path, mode="a", header=write_header, index=False)

    def start_simulation(self):
        try:
            iterations = int(self.iter_input.get())
            if iterations <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid positive number for iterations.")
            return

        # Prepare GUI for run
        self.run_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="⏸ Pause")
        self.stop_btn.config(state="normal")
        self.progress['value'] = 0
        self.status_text.set("Initializing thread...")
        self.result_box.delete(1.0, tk.END)

        self.pause_event.clear()
        self.stop_event.clear()

        # Get the current selection from Radiobuttons
        selected_file = self.load_choice.get()

        # Start simulation in background thread
        t = threading.Thread(
            target=run_monte_carlo_thread,
            args=(self, iterations, selected_file)
        )
        t.daemon = True  # Ensure thread closes if app closes
        self.sim_thread = t
        t.start()

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.config(text="Pause")
            self.update_status("Resuming simulation...")
        else:
            self.pause_event.set()
            self.pause_btn.config(text="▶ Resume")
            self.update_status("Simulation paused.")

    def stop_simulation(self):
        self.stop_event.set()
        self.pause_event.clear()
        self.pause_btn.config(state="disabled", text="Pause")
        self.stop_btn.config(state="disabled")
        self.update_status("Stopping simulation...")


if __name__ == "__main__":
    root = tk.Tk()
    app = ReliabilityGUI(root)
    root.mainloop()
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- File Paths (no input prompt) ---
GEN_DATA_FILE = "CEB_GEN_Each_unit_Master_data.csv"

# Defining two different load datasets
LOAD_FILE_A = "SRILANKAN_LOAD_CURVE_MODIFIED.csv"
LOAD_FILE_B = "../data/SRILANKAN_LOAD_CURVE_MODIFIED_2025.csv"

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "SMCS_Reliability_result.csv")
HOURS_PER_YEAR = 8760


# --- Data Loading Function ---
def load_data(load_path):
    """Loads generator and load data from CSVs."""
    try:
        df_gen = pd.read_csv(GEN_DATA_FILE, header=0)
        Gen_df = df_gen[['Unit Capacity (MW)', 'MTTR (hours)', 'MTTF (hours)']].astype(float)
        Gen = Gen_df['Unit Capacity (MW)'].values
        MTTF = Gen_df['MTTF (hours)'].values
        MTTR = Gen_df['MTTR (hours)'].values

        df_load = pd.read_csv(load_path, header=0)
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
def run_smcs_thread(gui, num_years, selected_load_path):
    Gen, MTTF, MTTR, Annual_Load_Profile = load_data(selected_load_path)
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
        if gui.stop_event.is_set():
            gui.update_status("Simulation stopped by user.")
            break

        while gui.pause_event.is_set() and not gui.stop_event.is_set():
            gui.update_status("Simulation paused.")
            gui.root.after(200)

        if gui.stop_event.is_set():
            gui.update_status("Simulation stopped by user.")
            break
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
    duration = datetime.now() - start_time
    if current_time > 0 and not gui.stop_event.is_set():
        LOLP = total_LOL_hours / total_study_hours
        LOLE = total_LOL_hours / num_years
        gui.show_results(total_LOL_hours, LOLP, LOLE, duration, num_years, selected_load_path)
    else:
        gui.run_btn.config(state="normal")
        gui.pause_btn.config(state="disabled")
        gui.stop_btn.config(state="disabled")


# --- Tkinter GUI ---
class SMCS_GUI:
    def __init__(self, root):
        self.root = root
        root.title("Sequential Monte Carlo CEB Reliability Simulator")
        root.geometry("750x650")
        root.configure(bg="#f0f4f8")

        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.sim_thread = None

        # Title Label
        tk.Label(
            root,
            text="Power System Reliability Simulator",
            font=("Poppins", 20, "bold"),
            fg="#005088"
        ).pack(pady=(20, 5))

        tk.Label(
            root,
            text="Sequential Monte Carlo (SMCS) Method",
            font=("Poppins", 12, "italic")
        ).pack(pady=(0, 20))

        # -----------------------------
        # INPUT: NUM YEARS
        # -----------------------------
        input_frame = tk.LabelFrame(root, text=" Simulation Settings ", font=("Poppins", 10, "bold"), padx=20, pady=10, bg="#f0f4f8")
        input_frame.pack(pady=10, fill="x", padx=40)

        tk.Label(input_frame, text="Enter Number of Years:", font=("Poppins", 11), bg="#f0f4f8").grid(row=0, column=0, sticky="w")
        self.year_input = tk.Entry(input_frame, width=20, font=("Consolas", 11))
        self.year_input.insert(0, "10000")
        self.year_input.grid(row=0, column=1, padx=10)

        # -----------------------------
        # DATASET SELECTION
        # -----------------------------
        tk.Label(input_frame, text="Select Load Dataset:", font=("Poppins", 11), bg="#f0f4f8").grid(row=1, column=0, sticky="w",
                                                                                                      pady=(15, 0))

        self.load_choice = tk.StringVar(value=LOAD_FILE_A)

        rb1 = tk.Radiobutton(input_frame, text="Base Load Profile (Modified) only thermal & hydro", variable=self.load_choice,
                             value=LOAD_FILE_A, font=("Poppins", 10), bg="#f0f4f8")
        rb1.grid(row=1, column=1, sticky="w", pady=(15, 0))

        rb2 = tk.Radiobutton(input_frame, text="Actual Load Profile Both Conv+NonConv", variable=self.load_choice,
                             value=LOAD_FILE_B, font=("Poppins", 10), bg="#f0f4f8")
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
            font=("Poppins", 10),
            bg="#f0f4f8"
        )
        save_check.grid(row=3, column=0, sticky="w", pady=(15, 0))

        csv_entry = tk.Entry(input_frame, textvariable=self.csv_path_var, width=40, font=("Consolas", 10))
        csv_entry.grid(row=3, column=1, sticky="w", pady=(15, 0))

        # Progress Bar
        self.progress = ttk.Progressbar(root, length=650, mode='determinate')
        self.progress.pack(pady=15)

        # Status
        self.status_text = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status_text, font=("Poppins", 12), bg="#f0f4f8").pack(pady=5)

        # Live LOLP
        self.lolp_label = tk.StringVar(value="Live LOLP: -")
        tk.Label(root, textvariable=self.lolp_label, font=("Poppins", 12), bg="#f0f4f8").pack()

        # Start Button
        style = ttk.Style()
        style.configure("TButton", font=("Poppins", 12, "bold"))
        self.run_btn = ttk.Button(root, text="▶ Run Simulation", width=22, command=self.start_simulation)
        self.run_btn.pack(pady=8)

        control_frame = tk.Frame(root, bg="#f0f4f8")
        control_frame.pack(pady=(0, 12))

        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", width=12, command=self.toggle_pause, state="disabled")
        self.pause_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = ttk.Button(control_frame, text="■ Stop", width=12, command=self.stop_simulation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        # Results
        tk.Label(root, text="Final Results", font=("Poppins", 15, "bold"), bg="#f0f4f8").pack(pady=10)
        self.result_box = tk.Text(root, height=8, width=85, font=("Consolas", 10), bg="#f8fafc", padx=10, pady=10)
        self.result_box.pack()

    # GUI update functions
    def update_progress(self, current, total, lolp):
        pct = (current / total) * 10
        self.progress['value'] = pct
        self.lolp_label.set(f"Live LOLP: {lolp:.12f}")
        self.status_text.set(f"Running... {pct:.2f}% Completed")
        self.root.update_idletasks()

    def update_status(self, msg):
        self.status_text.set(msg)
        self.root.update_idletasks()

    def show_results(self, total_LOL_hours, LOLP, LOLE, duration, num_years, selected_load_path):
        self.progress['value'] = 100
        self.status_text.set("Simulation Completed ✔")
        self.result_box.delete(1.0, tk.END)
        self.result_box.insert(tk.END, f"{'=' * 60}\n")
        self.result_box.insert(tk.END, f" LOAD PROFILE: {os.path.basename(self.load_choice.get())}\n")
        self.result_box.insert(tk.END, f"{'=' * 60}\n")
        self.result_box.insert(tk.END, f" Total LOL Hours:               {total_LOL_hours:,.2f}\n")
        self.result_box.insert(tk.END, f" Final LOLP (Probability):      {LOLP:.12f}\n")
        self.result_box.insert(tk.END, f" Final LOLE (Hours/Year):       {LOLE:.2f}\n")
        duration_str = str(duration).split(".")[0]
        self.result_box.insert(tk.END, f" Computation Time:              {duration_str}\n")
        self.result_box.insert(tk.END, f"{'=' * 60}\n")

        if self.save_to_csv.get():
            try:
                self.save_results_csv(num_years, selected_load_path, total_LOL_hours, LOLP, LOLE, duration)
            except Exception as e:
                messagebox.showerror("CSV Save Error", f"Failed to save CSV:\n{e}")
                return
        messagebox.showinfo("Simulation Completed", "Sequential Monte Carlo Simulation Finished Successfully!")

    def save_results_csv(self, num_years, selected_load_path, total_LOL_hours, lolp, lole, duration):
        file_path = self.csv_path_var.get().strip() or RESULTS_CSV
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        load_type = "A" if selected_load_path == LOAD_FILE_A else "B" if selected_load_path == LOAD_FILE_B else os.path.basename(selected_load_path)
        row = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "years": int(num_years),
            "method": "SMCS",
            "load data type": load_type,
            "Total LOL Hours": f"{total_LOL_hours:.2f}",
            "LOLP": f"{lolp:.12f}",
            "LOLE": f"{lole:.12f}",
            "time taken": str(duration).split(".")[0],
        }

        df = pd.DataFrame([row])
        write_header = not os.path.exists(file_path)
        df.to_csv(file_path, mode="a", header=write_header, index=False)

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
        self.run_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="⏸ Pause")
        self.stop_btn.config(state="normal")

        self.pause_event.clear()
        self.stop_event.clear()

        selected_file = self.load_choice.get()

        t = threading.Thread(target=run_smcs_thread, args=(self, num_years, selected_file))
        t.daemon = True
        self.sim_thread = t
        t.start()

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.config(text="⏸ Pause")
            self.update_status("Resuming simulation...")
        else:
            self.pause_event.set()
            self.pause_btn.config(text="▶ Resume")
            self.update_status("Simulation paused.")

    def stop_simulation(self):
        self.stop_event.set()
        self.pause_event.clear()
        self.pause_btn.config(state="disabled", text="⏸ Pause")
        self.stop_btn.config(state="disabled")
        self.update_status("Stopping simulation...")


# --- Run GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = SMCS_GUI(root)
    root.mainloop()

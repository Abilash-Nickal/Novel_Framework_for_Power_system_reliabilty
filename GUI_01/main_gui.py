import tkinter as tk
from tkinter import messagebox
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# IMPORTANT: Import your custom logic from the engine file
import reliability_engine as engine


class ConvergenceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("SMCS-Reliability simulator (Convergence Monitor)")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f1f5f9")

        self.num_years = 2000  # Default study period
        self.update_queue = queue.Queue()

        # Load data through the engine
        try:
            self.data = engine.load_system_data(
                "../data/CEB_GEN_Each_unit_Master_data.csv",
                "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"
            )
        except Exception as e:
            messagebox.showerror("Data Error", f"Could not load CSV files: {e}")
            self.root.destroy()
            return

        self.setup_ui()
        self.root.after(100, self.check_queue)

    def setup_ui(self):
        # UI Styles and Layout
        header = tk.Frame(self.root, bg="#003b5c", height=70)
        header.pack(fill="x")
        tk.Label(header, text="SEQUENTIAL MONTE CARLO SIMULATION", fg="white", bg="#003b5c",
                 font=("Arial", 16, "bold")).pack(pady=15)

        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        self.btn = tk.Button(ctrl, text="RUN SIMULATION", bg="#10b981", fg="white", font=("Arial", 12, "bold"),
                             command=self.start_worker, padx=20)
        self.btn.pack(side="left", padx=10)

        self.progress_lbl = tk.Label(ctrl, text="Status: Ready", bg="white", font=("Arial", 10))
        self.progress_lbl.pack(side="left", padx=20)

        # Metrics Panels
        metrics = tk.Frame(self.root, bg="#f1f5f9")
        metrics.pack(fill="x", padx=20)

        self.lole_val = tk.StringVar(value="LOLE: 0.0000")
        self.year_val = tk.StringVar(value="Year: 0")

        tk.Label(metrics, textvariable=self.year_val, font=("Courier", 14, "bold"), bg="white", width=20, relief="flat",
                 pady=10).pack(side="left", padx=5)
        tk.Label(metrics, textvariable=self.lole_val, font=("Courier", 14, "bold"), bg="white", width=20, relief="flat",
                 pady=10).pack(side="left", padx=5)

        # Matplotlib Graph
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("LOLE Convergence Trend")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], color="#003b5c", linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def start_worker(self):
        self.btn.config(state="disabled")
        threading.Thread(target=self.simulation_thread, daemon=True).start()

    def simulation_thread(self):
        """This function handles the heavy lifting using the engine functions."""
        num_gen = len(self.data["Gen"])
        state = np.zeros(num_gen, dtype=int)
        time_to_event = -self.data["MTTF"] * np.log(np.random.rand(num_gen))

        total_lol_hours = 0.0
        current_time = 0.0
        total_target_hours = self.num_years * engine.HOURS_PER_YEAR
        last_reported_year = 0

        while current_time < total_target_hours:
            h_year = int(current_time) % engine.HOURS_PER_YEAR
            m_idx = self.data["month_lookup"][h_year]

            # CALL ENGINE FOR DISPATCH
            p_out, _ = engine.run_dispatch_step(state, self.data["Load"][h_year], m_idx, self.data)

            # Determine step size (dt) - jump to next event or 1 hour
            min_event_time = np.min(time_to_event)
            dt = min(1.0, min_event_time)

            if p_out < self.data["Load"][h_year] - 1e-4:
                total_lol_hours += dt

            current_time += dt
            time_to_event -= dt

            # State Transitions
            events = np.where(time_to_event <= 1e-6)[0]
            for i in events:
                state[i] = 1 - state[i]
                ref = self.data["MTTF"][i] if state[i] == 0 else self.data["MTTR"][i]
                time_to_event[i] = -ref * np.log(np.random.rand())

            # Update GUI every 10 simulated years
            cur_year = int(current_time // engine.HOURS_PER_YEAR)
            # Fix: Ensure cur_year > 0 to avoid ZeroDivisionError and check for new year interval
            if cur_year > last_reported_year and cur_year % 10 == 0:
                self.update_queue.put({"y": cur_year, "lole": total_lol_hours / cur_year})
                last_reported_year = cur_year

        self.update_queue.put({"done": True, "lole": total_lol_hours / self.num_years})

    def check_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()
                if "done" in msg:
                    self.btn.config(state="normal")
                    self.progress_lbl.config(text="Status: Complete ✔")
                    messagebox.showinfo("Result", f"Final LOLE: {msg['lole']:.4f} Hours/Year")
                else:
                    self.year_val.set(f"Year: {msg['y']}")
                    self.lole_val.set(f"LOLE: {msg['lole']:.4f}")
                    x, y = self.line.get_data()
                    self.line.set_data(np.append(x, msg['y']), np.append(y, msg['lole']))
                    self.ax.relim();
                    self.ax.autoscale_view();
                    self.canvas.draw()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = ConvergenceDashboard(root)
    root.mainloop()
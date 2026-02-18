import tkinter as tk
from tkinter import messagebox
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# IMPORTING YOUR SEPARATE ENGINE
import reliability_engine as engine


class ConvergenceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Group 04: Reliability Convergence Monitor")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f1f5f9")

        self.num_years = 2000  # Default study period
        self.update_queue = queue.Queue()

        # Load data through the engine module
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
        # Header Styling
        header = tk.Frame(self.root, bg="#003b5c", height=70)
        header.pack(fill="x")
        tk.Label(header, text="SEQUENTIAL MONTE CARLO SIMULATION", fg="white", bg="#003b5c",
                 font=("Arial", 16, "bold")).pack(pady=15)

        # Control Bar
        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        self.btn = tk.Button(ctrl, text="RUN SIMULATION", bg="#10b981", fg="white", font=("Arial", 12, "bold"),
                             command=self.start_sim_thread, padx=20, relief="flat", cursor="hand2")
        self.btn.pack(side="left", padx=10)

        self.progress_lbl = tk.Label(ctrl, text="Status: Ready", bg="white", font=("Arial", 10), fg="#64748b")
        self.progress_lbl.pack(side="left", padx=20)

        # Metrics Panels
        metrics = tk.Frame(self.root, bg="#f1f5f9")
        metrics.pack(fill="x", padx=20)

        self.year_val = tk.StringVar(value="Year: 0")
        self.lole_val = tk.StringVar(value="LOLE: 0.0000")

        for var in [self.year_val, self.lole_val]:
            tk.Label(metrics, textvariable=var, font=("Courier New", 14, "bold"), bg="white", width=22, relief="flat",
                     pady=12).pack(side="left", padx=5)

        # Live Plot Area
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.ax.set_title("LOLE Convergence Trend", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Duration (Years)")
        self.ax.set_ylabel("LOLE (Hours/Year)")
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.line, = self.ax.plot([], [], color="#003b5c", linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def start_sim_thread(self):
        """Disables button and starts the engine in a separate thread."""
        self.btn.config(state="disabled", bg="#94a3b8")
        self.progress_lbl.config(text="Status: Simulating...")

        # Start the engine logic in the background
        threading.Thread(target=self.worker_thread, daemon=True).start()

    def worker_thread(self):
        """Calls the separate engine logic."""
        # This completely delegates the loop to reliability_engine.py
        engine.run_full_sequential_simulation(self.num_years, self.data, self.update_queue)

    def check_queue(self):
        """Checks the queue for updates from the engine and refreshes the UI."""
        try:
            while True:
                msg = self.update_queue.get_nowait()

                if msg["done"]:
                    self.btn.config(state="normal", bg="#10b981")
                    self.progress_lbl.config(text="Status: Complete ✔")
                    messagebox.showinfo("Simulation Result",
                                        f"Final LOLE: {msg['lole']:.4f} Hours/Year\n"
                                        f"LOEE: {msg['loee']:.2f} MWh/Year")
                else:
                    # Update textual data
                    self.year_val.set(f"Year: {msg['y']}")
                    self.lole_val.set(f"LOLE: {msg['lole']:.4f}")

                    # Update plot data
                    x, y = self.line.get_data()
                    self.line.set_data(np.append(x, msg['y']), np.append(y, msg['lole']))

                    # Adjust view
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.canvas.draw()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = ConvergenceDashboard(root)
    root.mainloop()
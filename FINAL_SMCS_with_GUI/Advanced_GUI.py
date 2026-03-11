import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import reliability_engine as engine


class AdvancedSMCSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("SMCS Reliability Monitor - Group 04")
        self.root.geometry("1100x800")
        self.root.configure(bg="#f1f5f9")

        self.num_years = 2000
        self.update_queue = queue.Queue()

        try:
            self.data = engine.load_system_data(
                "../data/CEB_GEN_Each_unit_Master_data.csv",
                "../data/SRILANKAN_LOAD_CURVE_MODIFIED.csv"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Files not found: {e}")
            self.root.destroy()
            return

        self.setup_ui()
        self.root.after(100, self.check_queue)

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=60)
        header.pack(fill="x")
        tk.Label(header, text="RELIABILITY FRAMEWORK MONITOR (SMCS)", fg="white",
                 bg="#003b5c", font=("Arial", 14, "bold")).pack(pady=15)

        # Control Panel
        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        self.run_btn = tk.Button(ctrl, text="START SIMULATION", bg="#10b981", fg="white",
                                 font=("Arial", 11, "bold"), command=self.start_sim, padx=20)
        self.run_btn.pack(side="left", padx=10)

        self.status_lbl = tk.Label(ctrl, text="Status: Ready", bg="white", fg="#64748b")
        self.status_lbl.pack(side="left", padx=20)

        # TABS FOR VIEWS
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Tab 1: Graph View
        self.graph_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.graph_tab, text=" GRAPH CONVERGENCE ")

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("LOLE Convergence")
        self.ax.set_ylabel("Hours/Year")
        self.line, = self.ax.plot([], [], color="#003b5c", lw=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tab 2: Numeric View (Live Dashboard)
        self.numeric_tab = tk.Frame(self.notebook, bg="#f8fafc")
        self.notebook.add(self.numeric_tab, text=" NUMERIC DASHBOARD ")

        self.setup_numeric_view()

    def setup_numeric_view(self):
        container = tk.Frame(self.numeric_tab, bg="#f8fafc", pady=40)
        container.pack(expand=True)

        self.metrics = {
            "y": ["Year of Study", "Years"],
            "lole": ["Current LOLE", "Hrs/Yr"],
            "lolp": ["Current LOLP", "Probability"],
            "loee": ["Current LOEE", "MWh/Yr"],
            "events": ["LOL Events", "Count"],
            "cost": ["Avg Annual Cost", "LKR"],
            "sim_time": ["Wall-clock Time", "Seconds"]
        }
        self.vars = {}

        # Create Grid for Metrics
        for i, (key, info) in enumerate(self.metrics.items()):
            row, col = divmod(i, 2)
            card = tk.Frame(container, bg="white", relief="groove", bd=1, padx=20, pady=20)
            card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

            tk.Label(card, text=info[0], bg="white", font=("Arial", 10), fg="#94a3b8").pack()

            var = tk.StringVar(value="0.00")
            self.vars[key] = var
            tk.Label(card, textvariable=var, bg="white", font=("Courier New", 22, "bold"), fg="#0f172a").pack(pady=5)
            tk.Label(card, text=info[1], bg="white", font=("Arial", 8, "italic"), fg="#64748b").pack()

    def start_sim(self):
        self.run_btn.config(state="disabled")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        engine.run_full_sequential_simulation(self.num_years, self.data, self.update_queue)

    def check_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()

                # Update Variables for Dashboard
                self.vars["y"].set(f"{msg['y']}")
                self.vars["lole"].set(f"{msg['lole']:.4f}")
                self.vars["lolp"].set(f"{msg['lolp']:.6f}")
                self.vars["loee"].set(f"{msg['loee']:.2f}")
                self.vars["events"].set(f"{msg['events']}")
                self.vars["cost"].set(f"{msg['cost'] / 1e6:.2f}M")
                self.vars["sim_time"].set(f"{msg['sim_time']:.1f}s")

                if not msg["done"]:
                    # Update Graph
                    x, y = self.line.get_data()
                    self.line.set_data(np.append(x, msg['y']), np.append(y, msg['lole']))
                    self.ax.relim();
                    self.ax.autoscale_view();
                    self.canvas.draw()
                else:
                    self.run_btn.config(state="normal")
                    self.status_lbl.config(text="Status: Complete ✔")
                    messagebox.showinfo("Done", "Study Completed Successfully.")
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedSMCSGui(root)
    root.mainloop()
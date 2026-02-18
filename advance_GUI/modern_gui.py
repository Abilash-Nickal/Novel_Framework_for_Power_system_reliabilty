import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import reliability_engine_2 as engine


class AdvancedSMCSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("SMCS Reliability Monitor - Group 04")
        self.root.geometry("1300x850")
        self.root.configure(bg="#f1f5f9")

        self.num_years = 2000
        self.update_queue = queue.Queue()

        # State for plotting
        self.current_metric = "lole"
        self.history = {
            "y": [],
            "lole": [],
            "lolp": [],
            "loee": [],
            "cost": []
        }

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
        # Apply ttk theme for a more modern look
        self.style = ttk.Style()
        self.style.theme_use('default') # 'clam', 'alt', 'default', 'classic' options

        # Custom styles for buttons to simulate modern feel/hover
        self.style.configure('TButton',
                             font=('Arial', 10),
                             padding=5,
                             relief='flat',
                             background='#e0e0e0', # Default background
                             foreground='#333333'
                            )
        self.style.map('TButton',
                       background=[('active', '#cccccc'), # Hover effect
                                   ('disabled', '#f0f0f0')],
                       foreground=[('disabled', '#a0a0a0')]
                      )

        # Style for the main run button
        self.style.configure('Run.TButton',
                             background='#10b981',
                             foreground='white',
                             font=('Arial', 11, 'bold')
                            )
        self.style.map('Run.TButton',
                       background=[('active', '#0e9c70'), # Darker green on hover
                                   ('disabled', '#94a3b8')],
                       foreground=[('disabled', 'white')]
                      )

        # Style for active metric buttons
        self.style.configure('Active.TButton',
                             background='#003b5c',
                             foreground='white',
                             font=('Arial', 9, 'bold')
                            )
        self.style.map('Active.TButton',
                       background=[('active', '#002f4a')]
                      )

        # Style for inactive metric buttons
        self.style.configure('Inactive.TButton',
                             background='#f1f5f9',
                             foreground='#0f172a',
                             font=('Arial', 9)
                            )
        self.style.map('Inactive.TButton',
                       background=[('active', '#e0e0e0')]
                      )


        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=60)
        header.pack(fill="x")
        tk.Label(header, text="RELIABILITY FRAMEWORK MONITOR (SMCS)", fg="white",
                 bg="#003b5c", font=("Arial", 16, "bold")).pack(pady=15)

        # Control Panel (Top)
        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        # Use ttk.Button for run_btn
        self.run_btn = ttk.Button(ctrl, text="START SIMULATION", style='Run.TButton',
                                 command=self.start_sim, cursor="hand2")
        self.run_btn.pack(side="left", padx=10)

        self.status_lbl = tk.Label(ctrl, text="Status: Ready", bg="white", font=("Arial", 10), fg="#64748b")
        self.status_lbl.pack(side="left", padx=20)

        # Main Content Area (Side-by-Side)
        main_content = tk.Frame(self.root, bg="#f1f5f9")
        main_content.pack(fill="both", expand=True, padx=20, pady=5)

        # LEFT PANEL: Graphs
        left_panel = tk.Frame(main_content, bg="white", relief="flat", bd=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Graph Title & Canvas
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.ax.set_title("LOLE Convergence", fontsize=12, fontweight='bold')
        self.ax.set_ylabel("Hours/Year")
        self.line, = self.ax.plot([], [], color="#003b5c", lw=2)
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
            # Use ttk.Button for metric switcher buttons
            btn = ttk.Button(switch_frame, text=label, command=lambda k=key: self.change_graph(k),
                            cursor="hand2")
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
            "y": ["Year of Study", "Years"],
            "lole": ["Current LOLE", "Hrs/Yr"],
            "lolp": ["Current LOLP", "Probability"],
            "loee": ["Current LOEE", "MWh/Yr"],
            "events": ["LOL Events", "Count"],
            "cost": ["Avg Annual Cost", "LKR"],
            "sim_time": ["Simulation Time", "Seconds"]
        }
        self.vars = {}

        for key, info in self.metrics_info.items():
            card = tk.Frame(scroll_container, bg="white", relief="flat", bd=0, padx=15, pady=12)
            card.pack(fill="x", pady=(0, 10))

            tk.Label(card, text=info[0].upper(), bg="white", font=("Arial", 8, "bold"), fg="#94a3b8").pack(anchor="w")

            var = tk.StringVar(value="0.00")
            self.vars[key] = var
            tk.Label(card, textvariable=var, bg="white", font=("Courier New", 18, "bold"), fg="#0f172a").pack(
                anchor="w", pady=2)
            tk.Label(card, text=info[1], bg="white", font=("Arial", 8, "italic"), fg="#cbd5e1").pack(anchor="w")

    def change_graph(self, metric_key):
        self.current_metric = metric_key
        self.update_button_styles()

        # Update Titles and Labels
        titles = {"lole": "LOLE Convergence", "lolp": "LOLP Convergence", "loee": "LOEE Convergence",
                  "cost": "Annual Cost Convergence"}
        ylabels = {"lole": "Hours/Year", "lolp": "Probability", "loee": "MWh/Year", "cost": "LKR"}

        self.ax.set_title(titles[metric_key], fontsize=12, fontweight='bold')
        self.ax.set_ylabel(ylabels[metric_key])

        # Refresh plot with existing history
        if self.history["y"]:
            self.line.set_data(self.history["y"], self.history[self.current_metric])
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas.draw()

    def update_button_styles(self):
        for key, btn in self.switch_btns.items():
            if key == self.current_metric:
                btn.config(style='Active.TButton')
            else:
                btn.config(style='Inactive.TButton')

    def start_sim(self):
        # Configure the ttk.Button style for disabled state
        self.run_btn.config(state="disabled", style='Run.TButton')
        self.status_lbl.config(text="Status: Simulating...")
        # Reset history
        for key in self.history: self.history[key] = []
        self.line.set_data([], [])

        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        engine.run_full_sequential_simulation(self.num_years, self.data, self.update_queue)

    def check_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()

                # Update Variables
                self.vars["y"].set(f"{msg['y']}")
                self.vars["lole"].set(f"{msg['lole']:.4f}")
                self.vars["lolp"].set(f"{msg['lolp']:.6f}")
                self.vars["loee"].set(f"{msg['loee']:.2f}")
                self.vars["events"].set(f"{msg['events']}")
                self.vars["cost"].set(f"{msg['cost'] / 1e6:.2f}M")
                self.vars["sim_time"].set(f"{msg['sim_time']:.1f}s")

                if not msg["done"]:
                    # Update History for all metrics
                    self.history["y"].append(msg['y'])
                    self.history["lole"].append(msg['lole'])
                    self.history["lolp"].append(msg['lolp'])
                    self.history["loee"].append(msg['loee'])
                    self.history["cost"].append(msg['cost'])

                    # Update active graph
                    self.line.set_data(self.history["y"], self.history[self.current_metric])
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.canvas.draw()
                else:
                    self.run_btn.config(state="normal", style='Run.TButton') # Restore normal style
                    self.status_lbl.config(text="Status: Complete ✔")
                    messagebox.showinfo("Done", "Reliability Study Completed.")
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedSMCSGui(root)
    root.mainloop()
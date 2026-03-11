import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# IMPORT THE ENGINE
import nsmcs_engine2 as engine


class NSMCSAdvancedGui:
    def __init__(self, root):
        self.root = root
        self.root.title("NSMCS Reliability Simulator")
        self.root.geometry("1300x850")
        self.root.configure(bg="#f1f5f9")

        # Cross-thread communication & controls
        self.update_queue = queue.Queue(maxsize=10)
        self.pause_event = threading.Event()  # Replaces the simple boolean for safe cross-module pausing

        self.current_metric = "lolp"
        self.history = {"n": [], "lolp": [], "lole": []}
        self.latest_failed_states = []

        # Load data through the engine
        try:
            self.data = engine.load_system_data(
                "../data/CEB_GEN_Each_unit_Master_data.csv",
                "../data/SRILANKAN_LOAD_CURVE_MODIFIED_2025.csv"
            )
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load datasets:\n{e}")
            self.root.destroy()
            return

        self.setup_ui()
        self.root.after(50, self.check_queue)

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=60)
        header.pack(fill="x")
        tk.Label(header, text="NON-SEQUENTIAL MONTE CARLO SIMULATION (NSMCS)", fg="white", bg="#003b5c",
                 font=("Arial", 16, "bold")).pack(pady=15)

        # Control Panel
        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        tk.Label(ctrl, text="Iterations:", font=("Arial", 10, "bold"), bg="white").pack(side="left", padx=(10, 5))
        self.iter_entry = tk.Entry(ctrl, font=("Consolas", 12), width=12)
        self.iter_entry.insert(0, "1000000")  # Default 1 Million
        self.iter_entry.pack(side="left", padx=5)

        self.run_btn = tk.Button(ctrl, text="▶ START SIMULATION", bg="#10b981", fg="white", font=("Arial", 10, "bold"),
                                 command=self.start_sim, padx=15, relief="flat", cursor="hand2")
        self.run_btn.pack(side="left", padx=10)

        self.pause_btn = tk.Button(ctrl, text="⏸ PAUSE", bg="#3b82f6", fg="white", font=("Arial", 10, "bold"),
                                   command=self.toggle_pause, padx=15, relief="flat", state="disabled", cursor="hand2")
        self.pause_btn.pack(side="left", padx=10)

        self.export_btn = tk.Button(ctrl, text="💾 EXPORT FAILED GENS", bg="#64748b", fg="white",
                                    font=("Arial", 10, "bold"), command=self.export_down_generators, padx=15,
                                    relief="flat", state="disabled", cursor="hand2")
        self.export_btn.pack(side="left", padx=10)

        self.status_lbl = tk.Label(ctrl, text="Status: Ready", bg="white", font=("Arial", 10), fg="#64748b")
        self.status_lbl.pack(side="left", padx=20)

        # Main Layout
        main_content = tk.Frame(self.root, bg="#f1f5f9")
        main_content.pack(fill="both", expand=True, padx=20, pady=5)

        # LEFT PANEL: Graphs
        left_panel = tk.Frame(main_content, bg="white")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Graph Switcher Buttons
        switch_frame = tk.Frame(left_panel, bg="white", pady=10)
        switch_frame.pack(fill="x")
        self.switch_btns = {}

        metrics_to_switch = [("LOLP (Probability)", "lolp"), ("LOLE (Hours/Year)", "lole"),
                             ("Grid State on Last Failure", "states")]
        for label, key in metrics_to_switch:
            btn = tk.Button(switch_frame, text=label, command=lambda k=key: self.change_graph(k),
                            font=("Arial", 9, "bold"), bg="#f1f5f9", relief="flat", padx=10, cursor="hand2")
            btn.pack(side="left", padx=5)
            self.switch_btns[key] = btn

        self.update_button_styles()
        self.redraw_graph()

        # RIGHT PANEL: Dashboard Stack
        right_panel = tk.Frame(main_content, bg="#f1f5f9", width=300)
        right_panel.pack(side="right", fill="y")
        self.vars = {}

        metrics_info = {
            "n": ["Iterations Completed", "Samples"],
            "lolp": ["Current LOLP", "Probability"],
            "lole": ["Current LOLE", "Hrs/Yr"],
            "events": ["Loss of Load Events", "Count"],
            "sim_time": ["Simulation Time", "Seconds"]
        }

        for key, info in metrics_info.items():
            card = tk.Frame(right_panel, bg="white", padx=15, pady=15)
            card.pack(fill="x", pady=(0, 10))
            tk.Label(card, text=info[0].upper(), bg="white", font=("Arial", 8, "bold"), fg="#94a3b8").pack(anchor="w")
            v = tk.StringVar(value="0") if key in ["n", "events"] else tk.StringVar(value="0.000")
            self.vars[key] = v
            tk.Label(card, textvariable=v, bg="white", font=("Courier New", 20, "bold"), fg="#0f172a").pack(anchor="w")

    # ---------------------------------------------------------
    # UI EVENT HANDLERS
    # ---------------------------------------------------------
    def toggle_pause(self):
        if not hasattr(self, 'sim_thread') or not self.sim_thread.is_alive(): return

        if self.pause_event.is_set():
            self.pause_event.clear()  # Resume
            self.pause_btn.config(text="⏸ PAUSE", bg="#3b82f6")
            self.status_lbl.config(text="Status: Simulating...")
        else:
            self.pause_event.set()  # Pause
            self.pause_btn.config(text="▶ RESUME", bg="#f59e0b")
            self.status_lbl.config(text="Status: Paused ⏸")

    def export_down_generators(self):
        if len(self.latest_failed_states) == 0:
            messagebox.showwarning("No Data", "No failures have occurred yet, or simulation hasn't started.")
            return

        # 1 = DOWN, 0 = UP
        down_indices = [i for i, state in enumerate(self.latest_failed_states) if state == 1]

        if not down_indices:
            messagebox.showinfo("Export Status", "No generators were down during the last check.")
            return

        export_data = {
            "Generator_Name": [self.data["Names"][i] for i in down_indices],
            "Capacity_MW": [self.data["Gen"][i] for i in down_indices],
            "Status": ["DOWN (Offline)"] * len(down_indices)
        }

        df_export = pd.DataFrame(export_data)
        iters = self.vars['n'].get().replace(',', '')
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"failed_state_iter_{iters}.csv",
            title="Save Offline Generators Data",
            filetypes=[("CSV Files", "*.csv")]
        )

        if file_path:
            df_export.to_csv(file_path, index=False)
            messagebox.showinfo("Success",
                                f"Exported {len(down_indices)} offline generators to CSV.\nTotal MW Lost: {sum(export_data['Capacity_MW']):.2f} MW")

    def change_graph(self, metric_key):
        self.current_metric = metric_key
        self.update_button_styles()
        self.redraw_graph()
        self.canvas.draw_idle()

    def redraw_graph(self):
        self.ax.clear()

        if self.current_metric == "states":
            if len(self.latest_failed_states) > 0:
                # 0 = UP (Green), 1 = DOWN (Black)
                colors = ['#10b981' if s == 0 else '#0f172a' for s in self.latest_failed_states]
                self.ax.bar(range(len(self.latest_failed_states)), 1, color=colors, width=1.0)

            self.ax.set_title("Grid State During Last Failure (Green = UP, Black = DOWN)", fontsize=11,
                              fontweight='bold')
            max_index = len(self.data["Gen"]) - 1
            self.ax.set_xlabel(f"Generator Index (0 to {max_index})")
            self.ax.set_yticks([])
            self.ax.set_xlim(-0.5, len(self.data["Gen"]) - 0.5)
        else:
            titles = {"lolp": "Loss of Load Probability (LOLP) Convergence",
                      "lole": "Loss of Load Expectation (LOLE) Convergence"}
            ylabels = {"lolp": "Probability", "lole": "Hours/Year"}

            if self.history["n"]:
                self.ax.plot(self.history["n"], self.history[self.current_metric], color="#003b5c", lw=2)

            self.ax.set_title(titles.get(self.current_metric, ""), fontsize=11, fontweight='bold')
            self.ax.set_ylabel(ylabels.get(self.current_metric, ""))
            self.ax.set_xlabel("Iterations")
            self.ax.grid(True, alpha=0.3, linestyle='--')
            self.ax.relim()
            self.ax.autoscale_view()

    def update_button_styles(self):
        for k, b in self.switch_btns.items():
            if k == self.current_metric:
                b.config(bg="#003b5c", fg="white")
            else:
                b.config(bg="#f1f5f9", fg="#0f172a")

    # ---------------------------------------------------------
    # LAUNCH ENGINE (RUNS IN BACKGROUND)
    # ---------------------------------------------------------
    def start_sim(self):
        try:
            target_iterations = int(self.iter_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of iterations.")
            return

        self.run_btn.config(state="disabled", bg="#94a3b8")
        self.pause_btn.config(state="normal")
        self.export_btn.config(state="normal")
        self.status_lbl.config(text="Status: Simulating...")
        self.iter_entry.config(state="disabled")

        # Reset trackers
        self.pause_event.clear()
        for k in self.history: self.history[k] = []
        self.latest_failed_states = []

        # Pass the separated engine function to the thread
        self.sim_thread = threading.Thread(
            target=engine.run_nsmcs_engine,
            args=(target_iterations, self.data, self.update_queue, self.pause_event),
            daemon=True
        )
        self.sim_thread.start()

    # ---------------------------------------------------------
    # QUEUE PROCESSOR
    # ---------------------------------------------------------
    def check_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()

                if not msg["done"]:
                    self.vars["n"].set(f"{msg['n']:,}")
                    self.vars["lolp"].set(f"{msg['lolp']:.6f}")
                    self.vars["lole"].set(f"{msg['lole']:.4f}")
                    self.vars["events"].set(f"{msg['events']:,}")
                    self.vars["sim_time"].set(f"{msg['sim_time']:.1f}s")

                    self.history["n"].append(msg["n"])
                    self.history["lolp"].append(msg["lolp"])
                    self.history["lole"].append(msg["lole"])

                    self.latest_failed_states = msg.get("states", [])

                    self.redraw_graph()
                    self.canvas.draw_idle()
                else:
                    self.run_btn.config(state="normal", bg="#10b981")
                    self.pause_btn.config(state="disabled")
                    self.iter_entry.config(state="normal")
                    self.status_lbl.config(text="Status: Complete ✔")
                    messagebox.showinfo("Done", f"Simulation Finished.\nFinal LOLE: {msg['lole']:.2f} Hrs/Yr")
                    break
        except queue.Empty:
            pass

        self.root.after(50, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = NSMCSAdvancedGui(root)
    root.mainloop()
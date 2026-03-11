import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import reliability_engine as engine


class AdvancedSMCSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("SMCS Reliability Monitor - Group 04")
        self.root.geometry("1300x850")
        self.root.configure(bg="#f1f5f9")

        self.num_years = 30000

        # maxsize=5 is a neat trick. It limits the queue. If the GUI stops reading (Pauses),
        # the engine thread will block (Pause) when the queue is full, without needing engine modifications!
        self.update_queue = queue.Queue(maxsize=5)

        self.current_metric = "lole"
        self.is_paused = False

        self.history = {"y": [], "lole": [], "lolp": [], "loee": [], "cost": []}
        self.latest_states = []  # Will store the live generator state array

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
        self.root.after(50, self.check_queue)

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#003b5c", height=60)
        header.pack(fill="x")
        tk.Label(header, text="RELIABILITY FRAMEWORK MONITOR (SMCS)", fg="white", bg="#003b5c",
                 font=("Arial", 16, "bold")).pack(pady=15)

        # Control Panel
        ctrl = tk.Frame(self.root, bg="white", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        self.run_btn = tk.Button(ctrl, text="▶ START SIMULATION", bg="#10b981", fg="white", font=("Arial", 11, "bold"),
                                 command=self.start_sim, padx=20, relief="flat", cursor="hand2")
        self.run_btn.pack(side="left", padx=10)

        self.pause_btn = tk.Button(ctrl, text="⏸ PAUSE", bg="#3b82f6", fg="white", font=("Arial", 11, "bold"),
                                   command=self.toggle_pause, padx=20, relief="flat", state="disabled", cursor="hand2")
        self.pause_btn.pack(side="left", padx=10)

        self.export_btn = tk.Button(ctrl, text="💾 EXPORT DOWN GENS", bg="#64748b", fg="white",
                                    font=("Arial", 11, "bold"), command=self.export_down_generators, padx=20,
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

        metrics_to_switch = [("LOLE", "lole"), ("LOLP", "lolp"), ("LOEE", "loee"), ("Annual Cost", "cost"),
                             ("Gen States", "states")]
        for label, key in metrics_to_switch:
            btn = tk.Button(switch_frame, text=label, command=lambda k=key: self.change_graph(k),
                            font=("Arial", 9, "bold"), bg="#f1f5f9", relief="flat", padx=10, cursor="hand2")
            btn.pack(side="left", padx=5)
            self.switch_btns[key] = btn

        self.update_button_styles()
        self.redraw_graph()  # Initial draw

        # RIGHT PANEL: Dashboard
        right_panel = tk.Frame(main_content, bg="#f1f5f9", width=300)
        right_panel.pack(side="right", fill="y")
        self.vars = {}
        metrics_info = {
            "y": ["Year of Study", "Years"], "lole": ["Current LOLE", "Hrs/Yr"],
            "lolp": ["Current LOLP", "Prob"], "loee": ["Current LOEE", "MWh/Yr"],
            "events": ["LOL Events", "Count"], "cost": ["Avg Cost", "LKR"],
            "sim_time": ["Sim Time", "Sec"]
        }
        for key, info in metrics_info.items():
            card = tk.Frame(right_panel, bg="white", padx=15, pady=12)
            card.pack(fill="x", pady=(0, 10))
            tk.Label(card, text=info[0].upper(), bg="white", font=("Arial", 8, "bold"), fg="#94a3b8").pack(anchor="w")
            v = tk.StringVar(value="0.00")
            self.vars[key] = v
            tk.Label(card, textvariable=v, bg="white", font=("Courier New", 18, "bold"), fg="#0f172a").pack(anchor="w")

    def toggle_pause(self):
        """Toggles the simulation pause state."""
        if not hasattr(self, 'sim_thread') or not self.sim_thread.is_alive():
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.config(text="▶ RESUME", bg="#f59e0b")
            self.status_lbl.config(text="Status: Paused ⏸")
        else:
            self.pause_btn.config(text="⏸ PAUSE", bg="#3b82f6")
            self.status_lbl.config(text="Status: Simulating...")

    def export_down_generators(self):
        """Exports currently down generators to a CSV file."""
        if len(self.latest_states) == 0:
            messagebox.showwarning("No Data", "Simulation hasn't started or no data is available.")
            return

        # Fetch generator names from original CSV, maintaining the Merit Order sort logic
        try:
            df_gen = pd.read_csv("../data/CEB_GEN_Each_unit_Master_data.csv")
            cost_col = 'Unit Cost (LKR/kWh)'
            if cost_col not in df_gen.columns:
                cost_col = df_gen.columns[df_gen.columns.str.contains('Cost', case=False)][0]
            df_gen = df_gen.sort_values(by=cost_col).reset_index(drop=True)
            names = df_gen['Name'].values if 'Name' in df_gen.columns else df_gen.iloc[:, 0].values
        except Exception:
            names = [f"Gen_Index_{i}" for i in range(len(self.latest_states))]

        caps = self.data["Gen"]

        # 1 = DOWN, 0 = UP
        down_indices = [i for i, state in enumerate(self.latest_states) if state == 1]

        if not down_indices:
            messagebox.showinfo("Export Status", "Excellent! No generators are currently down.")
            return

        export_data = {
            "Merit_Order_Index": down_indices,
            "Generator_Name": [names[i] for i in down_indices],
            "Capacity_MW": [caps[i] for i in down_indices],
            "Status": ["DOWN (Offline)"] * len(down_indices)
        }

        df_export = pd.DataFrame(export_data)

        current_year = self.vars['y'].get()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"offline_generators_year_{current_year}.csv",
            title="Save Offline Generators Data",
            filetypes=[("CSV Files", "*.csv")]
        )

        if file_path:
            df_export.to_csv(file_path, index=False)
            messagebox.showinfo("Success",
                                f"Exported {len(down_indices)} offline generators to CSV.\nTotal MW Lost: {sum(export_data['Capacity_MW']):.2f} MW")

    def change_graph(self, metric_key):
        """Triggered when a user clicks a switch button."""
        self.current_metric = metric_key
        self.update_button_styles()
        self.redraw_graph()
        self.canvas.draw_idle()

    def redraw_graph(self):
        """Clears the canvas and draws the active plot (Line vs Bar Chart)."""
        self.ax.clear()

        if self.current_metric == "states":
            # Bar Chart logic for Generator States
            if len(self.latest_states) > 0:
                # 0 = UP (Green), 1 = DOWN (Black)
                colors = ['#10b981' if s == 0 else '#0f172a' for s in self.latest_states]
                self.ax.bar(range(len(self.latest_states)), 1, color=colors, width=1.0)

            self.ax.set_title("Live Generator States (Green = UP, Black = DOWN)", fontsize=12, fontweight='bold')
            self.ax.set_xlabel("Generator Index (0 to 125, sorted by Merit Order)")
            self.ax.set_yticks([])  # Hide Y-axis numbers
            self.ax.set_xlim(-0.5, len(self.latest_states) - 0.5)
        else:
            # Line Plot logic for Convergence metrics
            titles = {"lole": "LOLE Convergence", "lolp": "LOLP Convergence", "loee": "LOEE Convergence",
                      "cost": "Annual Cost Convergence"}
            ylabels = {"lole": "Hours/Year", "lolp": "Probability", "loee": "MWh/Year", "cost": "LKR"}

            if self.history["y"]:
                self.ax.plot(self.history["y"], self.history[self.current_metric], color="#003b5c", lw=2)

            self.ax.set_title(titles.get(self.current_metric, ""), fontsize=12, fontweight='bold')
            self.ax.set_ylabel(ylabels.get(self.current_metric, ""))
            self.ax.set_xlabel("Duration (Years)")
            self.ax.grid(True, alpha=0.3, linestyle='--')
            self.ax.relim()
            self.ax.autoscale_view()

    def update_button_styles(self):
        for k, b in self.switch_btns.items():
            if k == self.current_metric:
                b.config(bg="#003b5c", fg="white")
            else:
                b.config(bg="#f1f5f9", fg="#0f172a")

    def start_sim(self):
        self.run_btn.config(state="disabled", bg="#94a3b8")
        self.pause_btn.config(state="normal")
        self.export_btn.config(state="normal")
        self.status_lbl.config(text="Status: Simulating...")

        for k in self.history: self.history[k] = []
        self.latest_states = []

        self.sim_thread = threading.Thread(
            target=lambda: engine.run_full_sequential_simulation(self.num_years, self.data, self.update_queue),
            daemon=True)
        self.sim_thread.start()

    def check_queue(self):
        # By not fetching from the queue when paused, the queue hits its maxsize (5).
        # This causes the engine's update_queue.put() to block, successfully pausing the simulation!
        if not self.is_paused:
            try:
                while True:
                    msg = self.update_queue.get_nowait()

                    if not msg["done"]:
                        # Update textual data
                        for k in ["y", "lole", "lolp", "loee", "events", "sim_time"]:
                            self.vars[k].set(f"{msg[k]:.4f}" if isinstance(msg[k], float) else f"{msg[k]}")
                        self.vars["cost"].set(f"{msg['cost'] / 1e6:.2f}M")

                        # Update histories
                        for k in ["y", "lole", "lolp", "loee", "cost"]:
                            self.history[k].append(msg[k])

                        # Update the live state array
                        self.latest_states = msg.get("states", [])

                        # Refresh active graph
                        self.redraw_graph()
                        self.canvas.draw_idle()
                    else:
                        self.run_btn.config(state="normal", bg="#10b981")
                        self.pause_btn.config(state="disabled")
                        self.status_lbl.config(text="Status: Complete ✔")
                        messagebox.showinfo("Done", "Study Completed Successfully.")
                        break
            except queue.Empty:
                pass

        self.root.after(50, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedSMCSGui(root)
    root.mainloop()
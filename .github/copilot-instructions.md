## Quick orientation for AI coding agents

This repository contains Monte Carlo simulations (sequential and non-sequential) to estimate Loss of Load Probability (LOLP) for the Sri Lanka system. The codebase is a mixture of runnable python scripts and Jupyter notebooks plus CSV data in `data/`.

Key locations
- `LOLP Seq MCS/` — sequential Monte Carlo implementations (e.g. `Srilankan_LOLP_using_SMCS_10M.py`, `SMCS.py`).
- `LOLP Non-seq MCS/` — non-sequential (state sampling) implementations (e.g. `NSMCS.py`, `01.py`).
- `data/` — CSV inputs (generator parameters, FOR, MTTF/MTTR, and 8760-hour load profiles). Examples: `CEB_GEN_Each_unit_Master_data.csv`, `CEB_GEN_MTTR_&_MTTF_for_each_unit.csv`, `SriLanka_Load_8760hr_repeat.csv`.
- Notebooks and exploratory scripts are in `basic logic/` and at repository root (e.g. `LOLP.ipynb`, `simulation.ipynb`).

What the AI should know (practical rules)
- Entry points are plain scripts invoked as `python <script>.py` (each file typically has an `if __name__ == "__main__"` runner). Use those as authoritative examples for program flow.
- Common dependencies: `numpy`, `pandas`. Assume these must be installed in the environment for code to run.
- Data is read from CSVs via `pandas.read_csv(...)`. Scripts expect specific column names (examples below). Don’t rename CSV files unless you update all call sites; prefer updating the script to accept an explicit path.

Data / schema patterns to reference
- Generator tables: columns vary across files. Typical column names include: `Capacity (MW)` or `Unit Capacity (MW)`, `MTTF (hours)`, `MTTR (hours)` or FOR (Forced Outage Rate). Verify the target script’s expected column names before editing code that reads them.
- Load profile: many scripts expect a single-column 8760-hour series in `SriLanka_Load_8760hr_repeat.csv`. Scripts will repeat shorter files (e.g., 24-hour profile) to fill 8760 hours.

Important code patterns and examples (copy/paste examples for edits)
- Time-to-event (SMCS) sampling: sampled as exponential using MTTF/MTTR:
  - Example (from `Srilankan_LOLP_using_SMCS_10M.py`):
    `time_to_next_event = -MTTF * np.log(np.random.rand(num_generators))`
- Vectorized availability sampling (NSMCS): use a pre-generated random matrix and boolean mask:
  - Example (from `NSMCS.py`):
    `outage_mask = random_gen_checks[n] > FOR_np`
    `availableGen = np.sum(Gen_np * outage_mask)`

Repository conventions and gotchas
- Inconsistent column headers and folder name casing exist (e.g. `DATA/` vs `data/`). On Windows this is fine; on case-sensitive systems update paths carefully.
- Many scripts use large default constants (e.g. `NUM_YEARS=100000` or `NUM_ITERATIONS=100000000`). For development and tests, lower these to something small (100–1000) to get fast feedback.
- Some scripts create placeholder CSVs if a file is missing — look for that behavior before overwriting data files.

Developer workflows & quick-run tips
- Local dev (PowerShell on Windows): run a script directly, e.g.
  `python "LOLP Seq MCS\\Srilankan_LOLP_using_SMCS_10M.py"`
- To iterate quickly: set `NUM_YEARS` / `NUM_ITERATIONS` to a small value, run, validate outputs, then scale up for long runs.
- When changing CSV-reading logic, add a small unit test or short-run script that loads the CSV and asserts expected column names/types.

When making edits: clear, low-risk rules for AI changes
- Prefer non-destructive edits: add optional parameters (e.g. `--data-path`) rather than renaming or moving large data files.
- Normalize CSV reads by explicitly selecting columns and casting types (`df[['Capacity (MW)', 'MTTF (hours)']].astype(float)`) to avoid runtime surprises.
- Keep numeric constants (simulation length, seed) as top-level module constants so they are easy to change for testing.

When to ask a human
- Ambiguous column names (e.g. Capacity vs Unit Capacity) or unit mismatches. Ask before renaming columns or changing units.
- If a long-run experiment is requested (large NUM_YEARS / NUM_ITERATIONS) — confirm compute/resource constraints.

Files to inspect for patterns or to reference in PRs
- `LOLP Seq MCS/Srilankan_LOLP_using_SMCS_10M.py` — canonical SMCS flow and time-to-event sampling.
- `LOLP Seq MCS/SMCS.py` — variant with placeholder-file logic for missing generator CSVs.
- `LOLP Non-seq MCS/NSMCS.py` — vectorized non-sequential sampling and bulk random generation.

If anything in this file is unclear, tell me which script or CSV you want covered and I will update the instructions to include exact column names and a short test harness.

-- End of repo-specific Copilot instructions
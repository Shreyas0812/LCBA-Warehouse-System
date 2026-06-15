# Setup Guide — New Machine

Steps to get the project running after cloning on a fresh system (Linux assumed for college machine).

---

## 1. Prerequisites

- Python 3.10 or later
- git

Check versions:
```bash
python3 --version
git --version
```

---

## 2. Clone the repo

```bash
git clone <your-repo-url>
cd LCBA_Warehouse_System
```

---

## 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. All subsequent commands assume the venv is active.

---

## 4. Install the package and dependencies

```bash
# Install the project itself (editable mode so imports resolve correctly)
pip install -e .

# Additional packages used by the experiment runner and plotting
pip install scipy pandas tqdm psutil
```

Full list of what ends up installed:
| Package | Used for |
|---------|----------|
| numpy | simulation numerics |
| pyyaml | config file parsing |
| matplotlib | plotting |
| networkx | communication graph |
| scipy | stats in analysis |
| pandas | results aggregation |
| tqdm | progress bars |
| psutil | machine info in experiment_config.json |

---

## 5. Verify the install

```bash
# Should print without errors
python -c "from integration.orchestrator import IntegrationOrchestrator; print('OK')"
```

---

## 6. Check available CPU cores

Before running parallel experiments, check the machine's core count to avoid oversubscription:

```bash
nproc                              # logical cores available to this process
nproc --all                        # total logical cores
lscpu | grep "Core(s) per socket"  # physical cores per socket
lscpu | grep "Socket(s)"           # number of sockets
```

Use **physical cores** (not logical/hyperthreaded) as your `--workers` value, since each simulation step is pure compute.

---

## 7. Run all experiments (recommended)

The easiest way to run the full thesis experiment suite is the provided shell script, which handles all maps sequentially with the correct config per map:

```bash
bash run_all_experiments.sh                  # full thesis run
bash run_all_experiments.sh --mode medium    # faster / initial results
bash run_all_experiments.sh --mode quick     # smoke test only
bash run_all_experiments.sh --workers 8      # override worker count (default: all cores)
```

The script:
1. Activates the venv automatically
2. Runs a smoke test first — aborts if it fails
3. Runs each map sequentially with the correct `--config` per the strategy below

Per-map strategy:

| Map | N | Config | Methods |
|-----|---|--------|---------|
| warehouse_small | 6 | `all` | LCBA + CBBA + SGA + DMCHBA, ss + batch |
| warehouse_large | 18 | `all` | LCBA + CBBA + SGA + DMCHBA, ss + batch |
| crossdock | 50 | `gcbba_dmchba` | LCBA + DMCHBA, ss + batch |
| kiva | 100 | `gcbba_dmchba` | LCBA + DMCHBA, ss + batch |
| kiva_large | 200 | `gcbba_dmchba` | LCBA + DMCHBA, ss + batch |
| shelf_aisle | 470 | `gcbba_dmchba` | LCBA + DMCHBA, ss + batch |

CBBA and SGA are only run on the two smallest maps (N=6, N=18) where they are computationally feasible for both steady-state and batch modes. DMCHBA (SOTA baseline) runs on all maps.

---

## 8. Run a single map manually

If you need to re-run one map or a specific config subset:

```bash
python experiments/run_experiments.py \
  --map gridworld_warehouse_small \
  --mode medium \
  --config all \
  --workers <N>
```

Use `--workers 0` to auto-detect and use all cores on a dedicated machine.

### Available `--config` options

| Flag | What runs |
|------|-----------|
| `all` | Everything — all methods, ss + batch |
| `ss_only` | Steady-state configs only (task_arrival_rate > 0) |
| `batch_only` | Batch configs only (initial_tasks > 0, rate = 0) |
| `gcbba_dmchba` | LCBA (all variants) + DMCHBA, ss + batch. No CBBA/SGA |
| `gcbba_only` | LCBA variants only (static + dynamic + sensitivity sweep), ss + batch |
| `baselines_only` | CBBA + SGA + DMCHBA, ss + batch |
| `static_only` | Static LCBA ss only |
| `dynamic_only` | Dynamic LCBA (canonical + ri sweep) ss only |
| `cbba_only` | CBBA ss only |
| `sga_only` | SGA ss only |
| `dmchba_only` | DMCHBA ss + batch |
| `sensitivity_only` | dynamic_ri* sweep configs only |

### Available `--mode` options

| Flag | Seeds | Arrival rates | Comm ranges | Batch task counts |
|------|-------|---------------|-------------|-------------------|
| `quick` | 1 | 2 | 2 | 1 |
| `medium` | 3 | 4 | 4 | 3 |
| `full` | 5 | 10 | 6 | 4 |

---

## 9. Results

Results are written to:
```
results/experiments/<map_name>/<timestamp>/
    experiment_config.json   ← run parameters + machine info
    summary.csv              ← one row per run
    summary_with_optimality.csv
    <run_id>/
        metrics.json
        trajectories.csv
```

---

## 10. Notes on wall-clock limits and parallelism

- The experiment runner caps each run at **600s wall-clock time** (`WALL_CLOCK_LIMIT_S`).
- This is measured in real elapsed time inside each worker process.
- On a dedicated machine, `--workers 0` (all cores) is safe and gives maximum parallelism. On a shared machine, keep `--workers ≤ physical_core_count` to avoid oversubscription inflating wall time.
- CBBA and SGA runs at high arrival rates are the slowest — they will most often hit the 600s cap on large maps. Runs that hit the cap are recorded with `hit_wall_clock_ceiling=True` in the CSV.

---

## 11. Keeping the venv active across sessions

The venv deactivates when the shell session ends. Re-activate it each time:

```bash
source .venv/bin/activate
```

Or add it to your shell rc file if you want it automatic:
```bash
echo "source ~/LCBA_Warehouse_System/.venv/bin/activate" >> ~/.bashrc
```

---

## Appendix: Why CBBA/SGA are excluded from large maps

On maps with N≥50 agents, CBBA and SGA hit the per-call allocation timeout (`allocation_timeout_s=10s`) repeatedly and can barely complete any simulation steps in batch mode. Their throughput approaches zero — not because they perform poorly on the task, but because they cannot compute an allocation fast enough to keep up with the simulation. This is a fundamentally different failure mode from "lower throughput".

For steady-state runs on kiva_large (N=200) and shelf_aisle (N=470), CBBA/SGA are also excluded via `gcbba_dmchba` since any data they produce is dominated by timeout artifacts rather than algorithmic behaviour.

DMCHBA is a distributed SOTA baseline and scales significantly better — it runs on all maps.

### Thesis framing

> "LCBA scales to N=470 agents with stable throughput. CBBA and SGA are computationally intractable at this scale, consistently exceeding the per-call allocation timeout of 10s, preventing meaningful simulation progress. DMCHBA, as a distributed SOTA baseline, is evaluated across all environments."
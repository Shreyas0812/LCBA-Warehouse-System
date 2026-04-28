"""
extract_summaries.py
====================
Reconstructs summary.csv from individual metrics.json files in an experiment directory.

Use this when an experiment completed successfully but summary.csv is empty or missing.

Usage:
    python experiments/extract_summaries.py --dir results/experiments/gridworld_crossdock/20260418_145313
    python experiments/extract_summaries.py --dir results/experiments/gridworld_crossdock/20260418_145313 --overwrite
"""

import argparse
import csv
import json
import os
import sys

COLUMNS = [
    "run_id", "config_name", "allocation_method", "path_planner", "experiment_type",
    "seed", "num_agents", "task_arrival_rate", "initial_tasks", "comm_range",
    "rerun_interval", "total_steps", "hit_timestep_ceiling", "hit_wall_clock_ceiling", "hit_saturation_ceiling", "stop_reason",
    "throughput", "throughput_per_agent", "avg_task_wait_time", "max_task_wait_time",
    "steady_state_tasks_completed", "total_tasks_injected", "tasks_dropped_by_queue_cap",
    "avg_queue_depth", "makespan", "all_tasks_completed", "num_tasks_completed",
    "num_tasks_total", "avg_allocation_time_ms", "avg_allocation_time_per_agent_ms",
    "num_allocation_calls", "max_allocation_time_ms", "std_allocation_time_ms",
    "avg_consensus_rounds_per_call", "total_consensus_rounds", "avg_idle_ratio",
    "task_balance_std", "distance_per_task", "total_distance_all_agents",
    "total_energy_consumed", "charging_time_fraction", "num_charging_events",
    "num_deadlocks", "wall_time_seconds",
]


def extract(exp_dir: str, overwrite: bool = False):
    out_path = os.path.join(exp_dir, "summary.csv")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not overwrite:
        print(f"summary.csv already exists and is non-empty. Use --overwrite to replace.")
        sys.exit(1)

    subdirs = sorted(
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    )

    rows = []
    skipped = []

    for subdir in subdirs:
        metrics_path = os.path.join(exp_dir, subdir, "metrics.json")

        if not os.path.exists(metrics_path):
            skipped.append((subdir, "no metrics.json"))
            continue

        if os.path.getsize(metrics_path) == 0:
            skipped.append((subdir, "empty metrics.json"))
            continue

        with open(metrics_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                skipped.append((subdir, f"invalid JSON: {e}"))
                continue

        rows.append({col: data.get(col) for col in COLUMNS})

    if not rows:
        print(f"No valid metrics.json files found in {exp_dir}. Nothing written.")
        if skipped:
            print(f"Skipped {len(skipped)} subdirectories:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")
        sys.exit(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} subdirectories:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to experiment run directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing non-empty summary.csv")
    args = parser.parse_args()
    extract(args.dir, overwrite=args.overwrite)
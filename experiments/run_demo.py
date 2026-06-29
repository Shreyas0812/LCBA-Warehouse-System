"""Steady-state LCBA demo run for website / README visuals.

Continuous task arrivals keep all robots perpetually ferrying for a fixed
horizon, producing a clean, loopable trajectories.csv (no winding-down tail).
Pair with experiments/animate_trajectory.py to render an MP4/GIF.

Environment overrides:
    DEMO_SEED       random seed              (default 42)
    DEMO_OUT        output directory         (default results/demo_warehouse_small)
    DEMO_TIMESTEPS  simulation horizon       (default 480)

Example:
    DEMO_SEED=2025 DEMO_TIMESTEPS=480 python experiments/run_demo.py
"""
import sys, os, random
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
import numpy as np

SEED = int(os.environ.get('DEMO_SEED', '42'))
np.random.seed(SEED); random.seed(SEED)

from experiments.run_single_experiment import MetricsOrchestrator

OUT = os.environ.get('DEMO_OUT', 'results/demo_warehouse_small')
os.makedirs(OUT, exist_ok=True)
HORIZON = int(os.environ.get('DEMO_TIMESTEPS', '480'))

orch = MetricsOrchestrator(
    config_path='config/gridworld_warehouse_small.yaml',
    task_arrival_rate=0.06,      # ~0.5 tasks/ts across 8 inducts -> busy, not saturated
    induct_queue_capacity=10,
    warmup_timesteps=0,
    initial_tasks=0,             # steady-state, not batch
    comm_range=11,
    rerun_interval=95,
    stuck_threshold=15,
    max_plan_time=200,
    allocation_method='gcbba',   # LCBA
    allocation_timeout_s=10.0,
    wall_clock_limit_s=2400.0,
    path_planner='ca_star',
    charger_planner='ca_star',
    idle_planner='ca_star',
    task_planner='rhcr',
)

print(f'Running {HORIZON} steady-state timesteps (seed {SEED})...')
orch.run_simulation(timesteps=HORIZON)

traj_path = os.path.join(OUT, 'trajectories.csv')
orch.save_trajectories(traj_path)
print(f'Completed tasks: {len(orch.completed_task_ids)}')
print(f'Trajectories -> {traj_path}')

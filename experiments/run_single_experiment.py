"""
run_single_experiment.py
========================
Single-run runners for the main thesis comparison experiments (GCBBA vs CBBA vs SGA).
Collects all RunMetrics fields.

See run_single_ri_sensitivity_experiment.py for the rerun_interval sensitivity runner.
"""

import os
import sys
import time
import random
from collections import OrderedDict

import numpy as np

from typing import Optional, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration.orchestrator import IntegrationOrchestrator
from Metrics import RunMetrics

class MetricsOrchestrator(IntegrationOrchestrator):
    """
    Extended instrumentation that collects all RunMetrics fields.
    Tracks allocation timing, consensus rounds, charging, deadlocks,
    and per-agent distance/energy.
    """

    def __init__(self, *args, **kwargs):
        self._wall_time_limit_s: Optional[float] = kwargs.pop("wall_clock_limit_s", None)
        self._saturation_stop_enabled: bool = kwargs.pop("saturation_stop_enabled", False)
        self._saturation_burn_in_steps: int = kwargs.pop("saturation_burn_in_steps", 300)
        self._saturation_window: int = kwargs.pop("saturation_window", 100)
        self._saturation_queue_frac: float = kwargs.pop("saturation_queue_frac", 0.9)
        self._saturation_backlog_growth_min: int = kwargs.pop("saturation_backlog_growth_min", 1)
        self._saturation_drop_growth_min: int = kwargs.pop("saturation_drop_growth_min", 1)
        self._saturation_consecutive_windows: int = kwargs.pop("saturation_consecutive_windows", 3)
        super().__init__(*args, **kwargs)

        self._hit_wall_clock_ceiling: bool = False
        self._hit_saturation_ceiling: bool = False
        self._saturation_consecutive_hits: int = 0

        # Allocation timing
        self._allocation_times_ms: List[float] = []
        self._allocation_call_timesteps: List[int] = []
        self._tasks_per_allocation_call: List[int] = []

        # Charging
        self._total_charging_timesteps: int = 0
        self._num_charging_events: int = 0
        self._prev_navigating: List[bool] = []   # populated on first step

        # Deadlocks (stuck-state entry events)
        self._num_deadlocks: int = 0
        self._prev_stuck: List[bool] = []        # populated on first step

        # Time series
        self._tasks_completed_over_time: List[int] = []
        self._pending_over_time: List[int] = []
        self._dropped_over_time: List[int] = []

    def _check_saturation_window(self) -> bool:
        if not self._saturation_stop_enabled:
            return False

        if self.current_timestep < self._saturation_burn_in_steps:
            return False

        if len(self._queue_depth_snapshots) < self._saturation_window:
            return False

        recent_q = self._queue_depth_snapshots[-self._saturation_window :]
        avg_recent_q = float(np.mean(recent_q)) if recent_q else 0.0
        queue_threshold = self._saturation_queue_frac * float(self.induct_queue_capacity)
        queue_near_cap = avg_recent_q >= queue_threshold

        if len(self._pending_over_time) <= self._saturation_window:
            pending_prev = self._pending_over_time[0]
        else:
            pending_prev = self._pending_over_time[-self._saturation_window - 1]
        pending_now = self._pending_over_time[-1]
        backlog_growth = pending_now - pending_prev

        if len(self._dropped_over_time) <= self._saturation_window:
            drops_prev = self._dropped_over_time[0]
        else:
            drops_prev = self._dropped_over_time[-self._saturation_window - 1]
        drops_now = self._dropped_over_time[-1]
        drop_growth = drops_now - drops_prev

        overload_signal = (
            backlog_growth >= self._saturation_backlog_growth_min
            or drop_growth >= self._saturation_drop_growth_min
        )
        return queue_near_cap and overload_signal
    
    def run_allocation(self) -> None:
        pending = len(self._pending_task_ids)
        t0 = time.perf_counter()
        IntegrationOrchestrator.run_allocation(self)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._allocation_times_ms.append(elapsed_ms)
        self._allocation_call_timesteps.append(self.current_timestep)
        self._tasks_per_allocation_call.append(pending)

    def run_simulation(self, timesteps: int = 100) -> None:
        from tqdm import tqdm
        wall_start = time.perf_counter()
        pbar = tqdm(range(timesteps), desc=f"Simulation ({self.allocation_method.upper()})", leave=True)
        for _ in pbar:
            if self._wall_time_limit_s is not None:
                if time.perf_counter() - wall_start > self._wall_time_limit_s:
                    self._hit_wall_clock_ceiling = True
                    tqdm.write(
                        f"[t={self.current_timestep}] WALL-CLOCK LIMIT "
                        f"({self._wall_time_limit_s / 60:.0f} min) exceeded — stopping early."
                    )
                    break
            self.step()

            if self._check_saturation_window():
                self._saturation_consecutive_hits += 1
                if self._saturation_consecutive_hits >= self._saturation_consecutive_windows:
                    self._hit_saturation_ceiling = True
                    tqdm.write(
                        f"[t={self.current_timestep}] SATURATION STOP "
                        f"(window={self._saturation_window}, burn_in={self._saturation_burn_in_steps}, "
                        f"queue>={self._saturation_queue_frac:.2f}*cap for "
                        f"{self._saturation_consecutive_windows} windows)."
                    )
                    break
            else:
                self._saturation_consecutive_hits = 0

            q        = float(np.mean(list(self._induct_queue_depth.values()))) if self._induct_queue_depth else 0
            active   = sum(1 for a in self.agent_states if not a.is_charging and not a.is_navigating_to_charger)
            nav_chg  = sum(1 for a in self.agent_states if a.is_navigating_to_charger)
            charging = sum(1 for a in self.agent_states if a.is_charging)
            pbar.set_postfix(OrderedDict([
                ("done",   len(self.completed_task_ids)),
                ("q",      f"{q:.2f}"),
                ("agents", f"{active}/{self.num_agents}"),
                ("nav",    f"{nav_chg}/{self.num_agents}"),
                ("chg",    f"{charging}/{self.num_agents}"),
                ("t",      self.current_timestep),
            ]), refresh=False)

    def step(self, *args, **kwargs):
        # Initialise per-agent state tracking on first step
        if not self._prev_navigating:
            self._prev_navigating = [s.is_navigating_to_charger for s in self.agent_states]
            self._prev_stuck = [s.is_stuck for s in self.agent_states]

        result = super().step(*args, **kwargs)

        for i, s in enumerate(self.agent_states):
            # New charging event: agent just started navigating to charger
            if s.is_navigating_to_charger and not self._prev_navigating[i]:
                self._num_charging_events += 1
            self._prev_navigating[i] = s.is_navigating_to_charger

            # Charging timesteps
            if s.is_charging:
                self._total_charging_timesteps += 1

            # Deadlock: agent just became stuck
            currently_stuck = s.detect_stuck(self.stuck_threshold)
            if currently_stuck and not self._prev_stuck[i]:
                self._num_deadlocks += 1
            self._prev_stuck[i] = currently_stuck

        self._tasks_completed_over_time.append(len(self.completed_task_ids))
        self._pending_over_time.append(len(self._pending_task_ids))
        self._dropped_over_time.append(int(self._tasks_dropped_by_cap))
        return result
    
    def collect_steady_state_metrics(self, warmup_timesteps: int, **kwargs) -> RunMetrics:
        """Collect metrics for a steady-state run (throughput, wait time, queue depth)."""
        m = RunMetrics()
        m.num_tasks_completed = len(self.completed_task_ids)
        self._collect_common_metrics(m, **kwargs)

        # ── Throughput ────────────────────────────────────────────
        m.total_tasks_injected = self._next_task_id
        m.tasks_dropped_by_queue_cap = self._tasks_dropped_by_cap
        if self._queue_depth_snapshots:
            m.avg_queue_depth = round(float(np.mean(self._queue_depth_snapshots)), 3)

        seen: set = set()
        ss_tasks = []
        for s in self.agent_states:
            for t in s.completed_tasks:
                if t.task_id not in seen and t.start_time is not None and t.start_time >= warmup_timesteps:
                    seen.add(t.task_id)
                    ss_tasks.append(t)
        ss_steps = max(1, self.current_timestep - warmup_timesteps)
        m.steady_state_tasks_completed = len(ss_tasks)
        m.throughput = round(len(ss_tasks) / ss_steps, 4)
        m.throughput_per_agent = round(m.throughput / kwargs["num_agents"], 6) if kwargs.get("num_agents") else 0.0

        wait_times = []
        for t in ss_tasks:
            inj = self._task_injection_time.get(t.task_id)
            if inj is not None and t.start_time is not None:
                wait_times.append(t.start_time - inj)
        if wait_times:
            m.avg_task_wait_time = round(float(np.mean(wait_times)), 2)
            m.max_task_wait_time = round(float(max(wait_times)), 2)

        return m
    
    def collect_batch_metrics(self, **kwargs) -> RunMetrics:
        """Collect metrics for a batch run (makespan, all_tasks_completed)."""
        m = RunMetrics()
        m.num_tasks_total = len(self.all_task_ids)
        m.num_tasks_completed = len(self.completed_task_ids)
        self._collect_common_metrics(m, **kwargs)

        # ── Batch completion ──────────────────────────────────────
        m.all_tasks_completed = self.completed_task_ids >= self.all_task_ids
        if m.all_tasks_completed:
            all_completions = [
                t.completion_time
                for s in self.agent_states
                for t in s.completed_tasks
                if t.completion_time is not None
            ]
            m.makespan = int(max(all_completions)) if all_completions else -1
        # solution_quality_ratio left at -1.0 (Hungarian not run)

        return m
    
    def _collect_common_metrics(
        self,
        m: RunMetrics,
        config_name: str,
        allocation_method: str,
        experiment_type: str,
        seed: int,
        num_agents: int,
        task_arrival_rate: float,
        initial_tasks: int,
        comm_range: float,
        rerun_interval: int,
        stuck_threshold: int,
        queue_max_depth: int,
        max_timesteps: int,
        wall_time: float,
        path_planner: str = "ca_star",
    ) -> None:
        """Fills fields common to both steady-state and batch runs."""

        # ── Identity ──────────────────────────────────────────────
        m.run_id = f"{config_name}_s{seed}"
        m.config_name = config_name
        m.allocation_method = allocation_method
        m.path_planner = path_planner
        m.experiment_type = experiment_type
        m.seed = seed
        m.num_agents = num_agents
        m.task_arrival_rate = task_arrival_rate
        m.initial_tasks = initial_tasks
        m.comm_range = comm_range
        m.rerun_interval = rerun_interval
        m.stuck_threshold = stuck_threshold
        m.queue_max_depth = queue_max_depth

        # ── Run validity ──────────────────────────────────────────
        m.total_steps = self.current_timestep
        m.hit_timestep_ceiling = (
            not (self.completed_task_ids >= self.all_task_ids)
            and self.current_timestep >= max_timesteps - 1
        )
        m.hit_wall_clock_ceiling = self._hit_wall_clock_ceiling
        m.hit_saturation_ceiling = self._hit_saturation_ceiling
        if m.hit_wall_clock_ceiling:
            m.stop_reason = "wall_clock_ceiling"
        elif m.hit_saturation_ceiling:
            m.stop_reason = "saturation_ceiling"
        elif m.hit_timestep_ceiling:
            m.stop_reason = "timestep_ceiling"
        else:
            m.stop_reason = "all_tasks_completed" if (self.completed_task_ids >= self.all_task_ids) else "completed"
        m.wall_time_seconds = round(wall_time, 3)

        # ── Computation ───────────────────────────────────────────
        m.num_allocation_calls = len(self._allocation_times_ms)
        if self._allocation_times_ms:
            m.total_allocation_time_ms = round(sum(self._allocation_times_ms), 2)
            m.avg_allocation_time_ms = round(float(np.mean(self._allocation_times_ms)), 2)
            m.avg_allocation_time_per_agent_ms = round(m.avg_allocation_time_ms / num_agents, 4) if num_agents else 0.0
            m.max_allocation_time_ms = round(float(max(self._allocation_times_ms)), 2)
            m.std_allocation_time_ms = round(float(np.std(self._allocation_times_ms)), 2)
        if self._tasks_per_allocation_call:
            m.avg_tasks_per_allocation_call = round(float(np.mean(self._tasks_per_allocation_call)), 2)

        # ── Communication ─────────────────────────────────────────
        m.total_consensus_rounds = self._total_consensus_rounds_all_calls
        if self._num_allocation_calls_with_consensus > 0:
            m.avg_consensus_rounds_per_call = round(
                self._total_consensus_rounds_all_calls / self._num_allocation_calls_with_consensus, 2
            )

        # ── Agent utilization ─────────────────────────────────────
        idle_ratios, tasks_per_agent = [], []
        for s in self.agent_states:
            hist = s.position_history
            total = max(len(hist), 1)
            idle = sum(1 for k in range(1, len(hist)) if hist[k - 1][:3] == hist[k][:3])
            idle_ratios.append(idle / total)
            tasks_per_agent.append(len(s.completed_tasks))
        m.per_agent_tasks_completed = tasks_per_agent
        m.avg_idle_ratio = round(float(np.mean(idle_ratios)), 4) if idle_ratios else 0.0
        m.std_idle_ratio = round(float(np.std(idle_ratios)), 4) if idle_ratios else 0.0
        m.task_balance_std = round(float(np.std(tasks_per_agent)), 3) if tasks_per_agent else 0.0

        # ── Distance ──────────────────────────────────────────────
        per_agent_dist = []
        for s in self.agent_states:
            hist = s.position_history
            dist = sum(1 for k in range(1, len(hist)) if hist[k - 1][:3] != hist[k][:3])
            per_agent_dist.append(float(dist))
        m.per_agent_distances = per_agent_dist
        m.total_distance_all_agents = round(sum(per_agent_dist), 1)
        m.avg_distance_per_agent = round(float(np.mean(per_agent_dist)), 2) if per_agent_dist else 0.0
        m.distance_per_task = round(m.total_distance_all_agents / max(m.num_tasks_completed, 1), 3)

        # ── Energy (1 move = 1 energy unit) ───────────────────────
        m.per_agent_energy_consumed = per_agent_dist
        m.total_energy_consumed = round(m.total_distance_all_agents, 1)
        m.avg_energy_consumed_per_agent = m.avg_distance_per_agent
        total_agent_timesteps = max(num_agents * self.current_timestep, 1)
        m.charging_time_fraction = round(self._total_charging_timesteps / total_agent_timesteps, 4)
        m.num_charging_events = self._num_charging_events
        m.total_charging_timesteps = self._total_charging_timesteps
        final_energies = [s.energy for s in self.agent_states]
        m.avg_final_energy = round(float(np.mean(final_energies)), 2)
        m.min_final_energy = int(min(final_energies))

        # ── Robustness ────────────────────────────────────────────
        m.num_deadlocks = self._num_deadlocks

        # ── Time series ───────────────────────────────────────────
        m.tasks_completed_over_time = self._tasks_completed_over_time
        m.allocation_call_timesteps = self._allocation_call_timesteps
        m.allocation_call_durations_ms = [round(x, 2) for x in self._allocation_times_ms]
        m.queue_depth_over_time = [round(x, 3) for x in self._queue_depth_snapshots]

    
# ─────────────────────────────────────────────────────────────────
#  Single experiment runners
# ─────────────────────────────────────────────────────────────────

def run_single_steady_state_experiment(
    config_path: str,
    config_name: str,
    task_arrival_rate: float,
    queue_max_depth: int,
    warmup_timesteps: int,
    comm_range: float,
    rerun_interval: int,
    stuck_threshold: int,
    seed: int,
    max_timesteps: int,
    allocation_method: str = "gcbba",
    initial_tasks: int = 0,
    allocation_timeout_s: Optional[float] = None,
    wall_clock_limit_s: Optional[float] = None,
    saturation_stop_enabled: bool = False,
    saturation_burn_in_steps: int = 300,
    saturation_window: int = 100,
    saturation_queue_frac: float = 0.9,
    saturation_backlog_growth_min: int = 1,
    saturation_drop_growth_min: int = 1,
    saturation_consecutive_windows: int = 3,
    max_plan_time: int = 200,
    path_planner: str = "ca_star",
    rhcr_replanning_period: int = None,
    charger_planner: Optional[str] = None,
    idle_planner: Optional[str] = None,
    task_planner: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RunMetrics:
    np.random.seed(seed)
    random.seed(seed)

    orch = MetricsOrchestrator(
        config_path=config_path,
        task_arrival_rate=task_arrival_rate,
        induct_queue_capacity=queue_max_depth,
        warmup_timesteps=warmup_timesteps,
        initial_tasks=initial_tasks,
        comm_range=comm_range,
        rerun_interval=rerun_interval,
        stuck_threshold=stuck_threshold,
        max_plan_time=max_plan_time,
        allocation_method=allocation_method,
        allocation_timeout_s=allocation_timeout_s,
        wall_clock_limit_s=wall_clock_limit_s,
        saturation_stop_enabled=saturation_stop_enabled,
        saturation_burn_in_steps=saturation_burn_in_steps,
        saturation_window=saturation_window,
        saturation_queue_frac=saturation_queue_frac,
        saturation_backlog_growth_min=saturation_backlog_growth_min,
        saturation_drop_growth_min=saturation_drop_growth_min,
        saturation_consecutive_windows=saturation_consecutive_windows,
        path_planner=path_planner,
        rhcr_replanning_period=rhcr_replanning_period,
        charger_planner=charger_planner,
        idle_planner=idle_planner,
        task_planner=task_planner,
    )

    t0 = time.perf_counter()
    orch.run_simulation(timesteps=max_timesteps)
    wall_time = time.perf_counter() - t0

    metrics = orch.collect_steady_state_metrics(
        warmup_timesteps=warmup_timesteps,
        config_name=config_name,
        allocation_method=allocation_method,
        path_planner=path_planner,
        experiment_type="steady_state",
        seed=seed,
        num_agents=len(orch.agent_states),
        task_arrival_rate=task_arrival_rate,
        initial_tasks=initial_tasks,
        comm_range=comm_range,
        rerun_interval=rerun_interval,
        stuck_threshold=stuck_threshold,
        queue_max_depth=queue_max_depth,
        max_timesteps=max_timesteps,
        wall_time=wall_time,
    )

    if output_dir is not None:
        run_dir = os.path.join(output_dir, metrics.run_id)
        os.makedirs(run_dir, exist_ok=True)
        orch.save_trajectories(os.path.join(run_dir, "trajectories.csv"))

    return metrics


def run_single_batch_experiment(
    config_path: str,
    config_name: str,
    initial_tasks: int,
    queue_max_depth: int,
    comm_range: float,
    rerun_interval: int,
    stuck_threshold: int,
    seed: int,
    max_timesteps: int,
    allocation_method: str = "gcbba",
    allocation_timeout_s: Optional[float] = None,
    wall_clock_limit_s: Optional[float] = None,
    max_plan_time: int = 200,
    path_planner: str = "ca_star",
    rhcr_replanning_period: int = None,
    charger_planner: Optional[str] = None,
    idle_planner: Optional[str] = None,
    task_planner: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> RunMetrics:
    np.random.seed(seed)
    random.seed(seed)

    orch = MetricsOrchestrator(
        config_path=config_path,
        task_arrival_rate=0.0,
        induct_queue_capacity=queue_max_depth,
        warmup_timesteps=0,
        initial_tasks=initial_tasks,
        comm_range=comm_range,
        rerun_interval=rerun_interval,
        stuck_threshold=stuck_threshold,
        max_plan_time=max_plan_time,
        allocation_method=allocation_method,
        allocation_timeout_s=allocation_timeout_s,
        wall_clock_limit_s=wall_clock_limit_s,
        path_planner=path_planner,
        rhcr_replanning_period=rhcr_replanning_period,
        charger_planner=charger_planner,
        idle_planner=idle_planner,
        task_planner=task_planner,
    )

    t0 = time.perf_counter()
    orch.run_simulation(timesteps=max_timesteps)
    wall_time = time.perf_counter() - t0

    metrics = orch.collect_batch_metrics(
        config_name=config_name,
        allocation_method=allocation_method,
        path_planner=path_planner,
        experiment_type="batch",
        seed=seed,
        num_agents=len(orch.agent_states),
        task_arrival_rate=0.0,
        initial_tasks=initial_tasks,
        comm_range=comm_range,
        rerun_interval=rerun_interval,
        stuck_threshold=stuck_threshold,
        queue_max_depth=queue_max_depth,
        max_timesteps=max_timesteps,
        wall_time=wall_time,
    )
    if metrics.all_tasks_completed:
        metrics.stop_reason = "all_tasks_completed"

    if output_dir is not None:
        run_dir = os.path.join(output_dir, metrics.run_id)
        os.makedirs(run_dir, exist_ok=True)
        orch.save_trajectories(os.path.join(run_dir, "trajectories.csv"))

    return metrics
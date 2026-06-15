from dataclasses import dataclass, field
from typing import List


@dataclass
class RunMetrics:
    """
    Per-run metrics collected for all allocation methods (LCBA, CBBA, SGA).

    Theoretical complexity per allocation call:
      LCBA   : O(n² · k)   n=agents, k=tasks
      CBBA   : O(n² · k)   same structure, different convergence constant
      SGA    : O(n · k)    single greedy pass, 1 round
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    run_id: str = ""
    config_name: str = ""
    allocation_method: str = ""       # "gcbba" | "cbba" | "sga"
    path_planner: str = "ca_star"     # "ca_star" | "rhcr"
    experiment_type: str = ""         # "steady_state" | "batch"
    seed: int = 0
    num_agents: int = 0
    task_arrival_rate: float = 0.0    # 0.0 for batch
    initial_tasks: int = 0            # 0 for steady-state
    comm_range: float = 0.0
    rerun_interval: int = 0
    stuck_threshold: int = 0
    queue_max_depth: int = 0

    # ── Run validity ───────────────────────────────────────────────────────────
    total_steps: int = 0
    hit_timestep_ceiling: bool = False
    hit_wall_clock_ceiling: bool = False
    hit_saturation_ceiling: bool = False
    stop_reason: str = ""   # "all_tasks_completed" | "timestep_ceiling" | "wall_clock_ceiling" | "saturation_ceiling"

    # ── Throughput (steady-state) ──────────────────────────────────────────────
    throughput: float = 0.0                   # PRIMARY — tasks/timestep post-warmup
    avg_task_wait_time: float = 0.0           # PRIMARY — injection → execution start (post-warmup)
    max_task_wait_time: float = 0.0           # secondary
    steady_state_tasks_completed: int = 0     # secondary — raw count (numerator of throughput)
    total_tasks_injected: int = 0             # secondary
    tasks_dropped_by_queue_cap: int = 0       # secondary — lost to queue overflow
    avg_queue_depth: float = 0.0              # secondary — mean pending tasks per station

    # ── Batch completion ───────────────────────────────────────────────────────
    makespan: int = -1                        # PRIMARY — timestep last task completed (-1 = N/A)
    all_tasks_completed: bool = False         # PRIMARY
    num_tasks_completed: int = 0              # secondary
    num_tasks_total: int = 0                  # secondary

    # ── Solution quality vs Hungarian (batch only) ─────────────────────────────
    # -1.0 = N/A (steady-state). 1.0 = matches optimal.
    solution_quality_ratio: float = -1.0     # PRIMARY — our_makespan / hungarian_makespan
    hungarian_makespan: int = -1              # secondary

    # ── Computation requirements ───────────────────────────────────────────────
    avg_allocation_time_ms: float = 0.0            # PRIMARY — per call
    avg_allocation_time_per_agent_ms: float = 0.0  # PRIMARY — per call / num_agents
    num_allocation_calls: int = 0                  # secondary
    total_allocation_time_ms: float = 0.0          # secondary
    max_allocation_time_ms: float = 0.0            # secondary
    std_allocation_time_ms: float = 0.0            # secondary
    num_allocation_timeouts: int = 0               # secondary
    avg_tasks_per_allocation_call: float = 0.0     # secondary — context: pending tasks per call

    # ── Communication requirements ─────────────────────────────────────────────
    # SGA has 0 rounds by definition. LCBA should converge in fewer rounds than CBBA.
    avg_consensus_rounds_per_call: float = 0.0     # PRIMARY
    total_consensus_rounds: int = 0                # secondary

    # ── Agent utilization & workload balance ──────────────────────────────────
    avg_idle_ratio: float = 0.0               # PRIMARY — fraction of timesteps agents are idle (mean)
    task_balance_std: float = 0.0             # PRIMARY — std of tasks completed per agent
    std_idle_ratio: float = 0.0               # secondary — spread across agents
    per_agent_tasks_completed: List[int] = field(default_factory=list)   # secondary (JSON only)

    # ── Path length / distance ─────────────────────────────────────────────────
    distance_per_task: float = 0.0            # PRIMARY — total_distance / tasks_completed
    total_distance_all_agents: float = 0.0    # secondary
    avg_distance_per_agent: float = 0.0       # secondary
    per_agent_distances: List[float] = field(default_factory=list)        # secondary (JSON only)

    # ── Energy ────────────────────────────────────────────────────────────────
    total_energy_consumed: float = 0.0           # PRIMARY
    charging_time_fraction: float = 0.0          # PRIMARY — charging timesteps / total agent-timesteps
    avg_energy_consumed_per_agent: float = 0.0   # secondary
    per_agent_energy_consumed: List[float] = field(default_factory=list)  # secondary (JSON only)
    num_charging_events: int = 0                 # secondary — times any agent navigated to charger
    total_charging_timesteps: int = 0            # secondary — agent-timesteps at charging station
    avg_final_energy: float = 0.0               # secondary
    min_final_energy: int = 0                   # secondary

    # ── Robustness ────────────────────────────────────────────────────────────
    num_deadlocks: int = 0                    # PRIMARY — distinct stuck-state entry events

    # ── Scalability ───────────────────────────────────────────────────────────
    throughput_per_agent: float = 0.0         # PRIMARY — throughput / num_agents

    # ── Wall-clock timing ─────────────────────────────────────────────────────
    wall_time_seconds: float = 0.0            # secondary

    # ── Time-series (JSON only, excluded from summary CSV) ────────────────────
    tasks_completed_over_time: List[int] = field(default_factory=list)
    allocation_call_timesteps: List[int] = field(default_factory=list)
    allocation_call_durations_ms: List[float] = field(default_factory=list)
    queue_depth_over_time: List[float] = field(default_factory=list)

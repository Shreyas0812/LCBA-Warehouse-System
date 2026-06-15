"""
Integration Orchestrator for Multi-Agent Task Allocation and Path Planning
- "gcbba" (default): LCBA — bundle building + local consensus on connected components
- "sga": Centralized Sequential Greedy Algorithm
- "cbba": Standard CBBA with FULLBUNDLE + local consensus
- "dmchba": Distributed Matching-by-Clones Hungarian (Samiei & Sun, IEEE T-RO 2024)

Path Planner: Priority-based Time-Expanded A* with reservation table for collision avoidance
"""

import os
import csv
import random
from typing import Optional, Set, Tuple, List, Dict
import yaml
import networkx as nx
import numpy as np
import time
from tqdm import tqdm
from dataclasses import dataclass

from path_planning.grid_map import GridMap
from path_planning.cooperative_astar import CooperativeAStar
from path_planning.rhcr_castar import RHCRCAStar
from path_planning.priority_based_search import PriorityBasedSearch

from gcbba.GCBBA_Orchestrator import GCBBA_Orchestrator
from baselines.SGA_Orchestrator import SGA_Orchestrator
from baselines.CBBA_Orchestrator import CBBA_Orchestrator
from baselines.DMCHBA_Orchestrator import DMCHBA_Orchestrator

from gcbba.tools_warehouse import agent_init, create_graph_with_range

from integration.agent_state import AgentState

@dataclass
class OrchestratorEvents:
    """Events that can occur during the simulation that the orchestrator needs to handle"""
    completed_task_ids: List[int]
    stuck_agent_ids: List[int]
    gcbba_rerun: bool

class IntegrationOrchestrator:
    """
    Main Integration Orchestrator - supports GCBBA, SGA, CBBA and DMCHBA allocation

    - Run allocation to get task assignments
    - Assignments are sent to AgentState for execution
    - Collision Avoidance called for Path Planning and Replanning
    - Step simulation forward and update AgentState with new positions and task statuses
    - Trigger GCBBA replanning at specified intervals or when certain conditions are met (e.g. task completion, new tasks added, etc.)
    """
    
    def __init__(self, 
                 config_path: str, 
                 task_arrival_rate: float = 0.1,
                 induct_queue_capacity: int = 5,
                 warmup_timesteps: int = 100,
                 initial_tasks: int = 0,
                 comm_range: float = 30,
                 sp_lim: Tuple[float, float] = (1.0, 1.0),
                 rerun_interval: int = 10,
                 new_task_cooldown: int = 5,
                 stuck_threshold: int = 15,
                 prediction_horizon: int = 5,
                 max_plan_time: int = 400,
                 Lt: Optional[int] = None,
                 allocation_method: str = "gcbba",  # "gcbba", "sga", "cbba" or "dmchba"
                 path_planner: str = "ca_star",          # "ca_star", "rhcr", or "pbs"
                 rhcr_replanning_period: int = None,    # h parameter; defaults to window_size (h=w)
                 allocation_timeout_s: Optional[float] = None,  # max seconds per allocation call; None = unlimited
                 idle_wait_after: int = 30,
                 no_path_replan_limit: int = 40,
                 stuck_task_release_after: int = 120,
                 charger_planner: Optional[str] = None,  # "ca_star", "rhcr", or "pbs" for charger paths; None=use path_planner
                 idle_planner: Optional[str] = None,     # "ca_star", "rhcr", or "pbs" for idle paths; None=use path_planner
                 task_planner: Optional[str] = None,     # "ca_star", "rhcr", or "pbs" for task paths; None=use path_planner
                 ) -> None:

        if allocation_method not in {"gcbba", "sga", "cbba", "dmchba"}:
            raise ValueError(f"Invalid allocation method: {allocation_method}. Must be one of 'gcbba', 'sga', 'cbba', or 'dmchba'.")

        self.allocation_method = allocation_method
        self.task_arrival_rate = task_arrival_rate
        self.induct_queue_capacity = induct_queue_capacity
        self.warmup_timesteps = warmup_timesteps
        self.initial_tasks = initial_tasks
        self.comm_range = comm_range
        self.sp_lim = sp_lim
        self.rerun_interval = rerun_interval
        self.new_task_cooldown = new_task_cooldown
        self.stuck_threshold = stuck_threshold
        self.prediction_horizon = prediction_horizon
        self.max_plan_time = max_plan_time
        self.Lt = Lt
        self.allocation_timeout_s = allocation_timeout_s
        self.idle_wait_after = idle_wait_after
        self.no_path_replan_limit = no_path_replan_limit
        self.stuck_task_release_after = stuck_task_release_after

        self.grid_map = GridMap(config_path)
        _planners = {"ca_star": CooperativeAStar, "rhcr": RHCRCAStar, "pbs": PriorityBasedSearch}
        if path_planner not in _planners:
            raise ValueError(f"Invalid path_planner: {path_planner!r}. Must be one of {list(_planners)}")
        if path_planner == "rhcr":
            self.path_planner = RHCRCAStar(self.grid_map, replanning_period=rhcr_replanning_period)
        else:
            self.path_planner = _planners[path_planner](self.grid_map)

        # Initialize phase-specific planners if provided; otherwise all phases use self.path_planner
        self.planner_map = None  # Will be set below if any phase-specific planner is specified
        if charger_planner or idle_planner or task_planner:
            # Extract the shared CA* reservation table from the default planner so all phase
            # planners write to the same table (Phase 2 routes around Phase 1, etc.)
            shared_ca = (
                self.path_planner._ca         # RHCRCAStar wraps a CooperativeAStar
                if isinstance(self.path_planner, RHCRCAStar)
                else self.path_planner        # CooperativeAStar IS the table
            )

            # At least one phase-specific planner was specified; create a map
            # Default each phase to path_planner if not explicitly overridden
            def _create_planner(planner_name: Optional[str]) -> object:
                if planner_name is None:
                    return self.path_planner
                if planner_name not in _planners:
                    raise ValueError(f"Invalid phase planner: {planner_name!r}. Must be one of {list(_planners)}")
                if planner_name == "rhcr":
                    return RHCRCAStar(self.grid_map, replanning_period=rhcr_replanning_period,
                                     shared_ca=shared_ca)
                else:
                    new_ca = _planners[planner_name](self.grid_map)
                    # Share the reservation dicts so all planners see each other's reservations
                    new_ca.reservations      = shared_ca.reservations
                    new_ca.goal_reservations = shared_ca.goal_reservations
                    return new_ca

            self.planner_map = {
                "charger": _create_planner(charger_planner),
                "idle":    _create_planner(idle_planner),
                "task":    _create_planner(task_planner),
            }

        (
            agent_positions,
            self.induct_positions,
            self.eject_positions,
            charging_positions,
            idle_task_positions,
            energy_config,
        ) = self._load_config(config_path)
        self.max_energy                  = energy_config['max_energy']
        self.charge_duration             = energy_config['charge_duration']
        self.charge_speed                 = energy_config['charge_speed']
        self.charging_trigger_multiplier = energy_config['charging_trigger_multiplier']
        self._validate_energy_config(agent_positions, charging_positions)

        # Inject Task Variables
        self._induct_last_injection: Dict[int, int] = {} # last timestep a task was injected at each induct station, keyed by induct station index
        self._induct_queue_depth: Dict[int, int] = {i: 0 for i in range(len(self.induct_positions))} # tracks how many tasks are currently queued or pending injection at each induct station, keyed by induct station index
        self._next_task_id: int = 0 # global task ID counter to ensure unique task IDs across the simulation
        self._task_to_induct: Dict[int, int] = {}      # task_id → induct station index
        self._task_injection_time: Dict[int, int] = {} # task_id → timestep injected
        self._pending_task_ids: Set[int] = set()       # tasks not yet claimed by any agent
        self._tasks_dropped_by_cap: int = 0            # count of tasks dropped due to induct queue capacity limits, for performance tracking and debugging
        self._queue_depth_snapshots: List[float] = []   # for tracking average queue depth over time, can be used for analysis and tuning of induct queue capacity and task arrival rates

        self._init_allocation(agent_positions)
        self._init_agent_states()
        self._stuck_task_steps: Dict[int, int] = {a.agent_id: 0 for a in self.agent_states}

        # Simulation state variables
        self.current_timestep = 0
        self.last_gcbba_timestep = -self.rerun_interval  # Initialize to allow GCBBA to run at timestep 0
        self.completed_task_ids: Set[int] = set()
        self._completed_at_last_gcbba: int = 0  # Track how many tasks were completed at the time of the last GCBBA run to help determine when to trigger next run

        self.latest_assignment: List[List[int]] = []  # Store latest GCBBA assignment for reference in stepping logic
        self._allocation_cancelled: bool = False  # Set by InstrumentedOrchestrator timeout to prevent zombie threads from mutating state
        self._force_allocation_rerun: bool = False

        # Consensus round accumulators
        self._total_consensus_rounds_all_calls: int = 0
        self._total_convergence_iterations_all_calls: int = 0
        self._num_allocation_calls_with_consensus: int = 0
        

        # For Energy and Charging Logic — use dedicated charging stations from config
        self.charging_station_grid_positions = [
            self.grid_map.continuous_to_grid(float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in charging_positions
        ]
        self.idle_task_grid_positions = [
            self.grid_map.continuous_to_grid(float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in idle_task_positions
        ]
        # Backward-compatible alias used by existing helper names.
        self.wait_station_grid_positions = self.idle_task_grid_positions
        self.station_grid_positions = {
            self.grid_map.continuous_to_grid(float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in (self.induct_positions + self.eject_positions)
        }
        self.eject_grid_positions = {
            self.grid_map.continuous_to_grid(float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in self.eject_positions
        }
        self.wait_forbidden_positions = set(self.station_grid_positions)
        for ex, ey, ez in self.eject_grid_positions:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_pos, ny_pos = ex + dx, ey + dy
                if self.grid_map.is_valid_cell(nx_pos, ny_pos, ez):
                    self.wait_forbidden_positions.add((nx_pos, ny_pos, ez))

    def _validate_energy_config(self, agent_positions: List, charging_positions: List) -> None:
        """
        Raise ValueError if any position an agent visits during task execution is so far
        from the nearest charger that charging_trigger_multiplier * distance > max_energy.
        Checks agent starting positions AND induct/eject stations, since the trigger fires
        based on current position during movement, not just the starting position.
        """
        if not charging_positions:
            return

        # All positions an agent may occupy when the charging check fires
        positions_to_check = (
            [(p[0], p[1], "agent") for p in agent_positions]
            + [(p[0], p[1], "induct") for p in self.induct_positions]
            + [(p[0], p[1], "eject") for p in self.eject_positions]
        )

        bad = []
        for x, y, kind in positions_to_check:
            dist = min(abs(x - cx) + abs(y - cy) for (cx, cy, *_) in charging_positions)
            if self.charging_trigger_multiplier * dist > self.max_energy:
                bad.append((x, y, kind, dist))

        if bad:
            required = max(self.charging_trigger_multiplier * d for *_, d in bad)
            lines = "\n".join(
                f"  {kind} at ({x},{y}): dist={d}, threshold={self.charging_trigger_multiplier*d:.0f}"
                for x, y, kind, d in bad
            )
            raise ValueError(
                f"Energy config error: {len(bad)} position(s) will perpetually trigger charging "
                f"(charging_trigger_multiplier={self.charging_trigger_multiplier} × distance > max_energy={self.max_energy}).\n"
                f"{lines}\n"
                f"Set max_energy >= {int(required) + 1} in the map YAML to fix this."
            )

    def _load_config(self, config_path: str) -> Tuple[List, List, List, List, List, Dict]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        params = config["create_gridworld_node"]["ros__parameters"]

        agent_pos_flat = params['agent_positions']
        agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], agent_pos_flat[i+3])
                                for i in range(0, len(agent_pos_flat), 4)]

        induct_pos_flat = params['induct_stations']
        induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3])
                                 for i in range(0, len(induct_pos_flat), 4)]

        eject_pos_flat = params['eject_stations']
        eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3])
                                for i in range(0, len(eject_pos_flat), 4)]

        charging_pos_flat = params.get('charging_stations', [])
        charging_positions = [(charging_pos_flat[i], charging_pos_flat[i+1], charging_pos_flat[i+2], charging_pos_flat[i+3])
                                for i in range(0, len(charging_pos_flat), 4)]

        wait_pos_flat = params.get('idle_task_stations', [])
        wait_positions = [(wait_pos_flat[i], wait_pos_flat[i+1], wait_pos_flat[i+2], wait_pos_flat[i+3])
                    for i in range(0, len(wait_pos_flat), 4)]

        energy_params = params.get('energy', {})
        energy_config = {
            'max_energy':                  int(energy_params.get('max_energy', 100)),
            'charge_duration':             int(energy_params.get('charge_duration', 20)),
            'charge_speed':                 int(energy_params.get('charge_speed', 1)),
            'charging_trigger_multiplier': float(energy_params.get('charging_trigger_multiplier', 2.0)),
        }

        return agent_positions, induct_positions, eject_positions, charging_positions, wait_positions, energy_config

    def _init_allocation(self, agent_positions: List) -> None:
        """
        Initialize the chosen allocation orchestrator (GCBBA, SGA, CBBA, or DMCHBA) with the given configuration.
        """
        raw_graph, G = create_graph_with_range(agent_positions, self.comm_range)
        if raw_graph.number_of_nodes() == 0:
            D = 1
        else:
            if nx.is_connected(raw_graph):
                D = nx.diameter(raw_graph)
            else:
                # If the graph is not fully connected, we can take the maximum diameter of the connected components as an approximation
                D = max(nx.diameter(raw_graph.subgraph(c)) for c in nx.connected_components(raw_graph))

        agents = agent_init(agent_positions, sp_lim=self.sp_lim)

        self.all_char_t: Dict[int, np.ndarray] = {}
        self.all_task_ids: Set[int] = set()
        self.all_char_a = agents   # List[np.array], indexed by agent index
        self.num_agents = len(agents)

        if self.task_arrival_rate > 0:
            arrival_interval = int(1.0 / self.task_arrival_rate)
            self._induct_last_injection = {i: -arrival_interval for i in range(len(self.induct_positions))}  # Initialize to allow immediate injection
        else:
            self._induct_last_injection = {
                i: 0 for i in range(len(self.induct_positions))
            }

        if self.initial_tasks > 0:
            for i in range(self.initial_tasks):
                induct_idx = i % len(self.induct_positions)
                induct_pos = self.induct_positions[induct_idx]
                eject_pos = self.eject_positions[np.random.randint(0, len(self.eject_positions))]
                char_t = np.array([induct_pos[0], induct_pos[1], induct_pos[2], eject_pos[0], eject_pos[1], eject_pos[2]])
                task_id = self._next_task_id
                
                self._next_task_id += 1
                self.all_char_t[task_id] = char_t
                self.all_task_ids.add(task_id)
                self._task_to_induct[task_id] = induct_idx
                self._task_injection_time[task_id] = 0  # Injected at timestep 0
                self._pending_task_ids.add(task_id)

        # Initialize using GCBBA -- all methods share same agent and task initialization, so we can use the same parameters to create the orchestrator instance and then call the appropriate launch method
        # tasks added on first step
        Lt = self.Lt if self.Lt is not None else 1
        self.gcbba_orchestrator_initial = GCBBA_Orchestrator(G, D, [], agents, Lt)

        # if self.task_arrival_rate > 0:
        #     print(f"Orchestrator initialized with {self.num_agents} agents. "
        #           f"Steady-state mode: arrival_rate={self.task_arrival_rate}/ts/station, "
        #           f"induct_queue_capacity={self.induct_queue_capacity}.")
        # else:
        #     print(f"Orchestrator initialized with {self.num_agents} agents. "
        #           f"Batch mode: {self.initial_tasks} pre-generated tasks, no ongoing injection.")

    def _init_agent_states(self) -> None:
        self.agent_states: List[AgentState] = []
        for gcbba_agent in self.gcbba_orchestrator_initial.agents:
            grid_pos = self.grid_map.continuous_to_grid(float(gcbba_agent.pos[0]), float(gcbba_agent.pos[1]), float(gcbba_agent.pos[2]))
            self.agent_states.append(AgentState(agent_id=gcbba_agent.agent_id, initial_position=grid_pos, speed=gcbba_agent.speed,
                                                 max_energy=self.max_energy, charge_speed=self.charge_speed,
                                                 no_current_task_threshold=self.idle_wait_after))

    def _inject_new_tasks(self) -> List[int]:
        """
        Deterministic periodic task injection: one task per induct station every
        arrival_interval timesteps. Tasks are skipped (counted as dropped) when the
        station's queue is already at induct_queue_capacity.
        Returns a list of newly injected task IDs.
        """
        if self.task_arrival_rate <= 0:
            return []  # No ongoing task injection in batch mode

        arrival_interval = max(1, round(1 / self.task_arrival_rate))
        new_task_ids = []

        for induct_idx, induct_pos in enumerate(self.induct_positions):
            if (self.current_timestep - self._induct_last_injection[induct_idx]) < arrival_interval:
                continue  # Not time to inject at this station yet
                
            self._induct_last_injection[induct_idx] = self.current_timestep

            if self._induct_queue_depth[induct_idx] >= self.induct_queue_capacity:
                self._tasks_dropped_by_cap += 1
                # tqdm.write(f"[t={self.current_timestep}] Induct station {induct_idx} queue full (depth={self._induct_queue_depth[induct_idx]}). Dropping new task. Total dropped: {self._tasks_dropped_by_cap}")
                continue  # Skip injection at this station due to capacity limit

            eject_pos = self.eject_positions[np.random.randint(0, len(self.eject_positions))]
            char_t = np.array([induct_pos[0], induct_pos[1], induct_pos[2], eject_pos[0], eject_pos[1], eject_pos[2]])
            task_id = self._next_task_id
            self._next_task_id += 1

            self.all_char_t[task_id] = char_t
            self.all_task_ids.add(task_id)
            self._task_to_induct[task_id] = induct_idx
            self._task_injection_time[task_id] = self.current_timestep
            self._induct_queue_depth[induct_idx] += 1
            self._pending_task_ids.add(task_id)
            new_task_ids.append(task_id)

        return new_task_ids

    def run_simulation(self, timesteps: int = 100) -> None:
        pbar = tqdm(range(timesteps), desc=f"Simulation ({self.allocation_method.upper()})", leave=True)
        for _ in pbar:
            events = self.step()
            done = len(self.completed_task_ids)
            q = float(np.mean(list(self._induct_queue_depth.values()))) if self._induct_queue_depth else 0
            active   = sum(1 for a in self.agent_states if not a.is_charging and not a.is_navigating_to_charger)
            nav_chg  = sum(1 for a in self.agent_states if a.is_navigating_to_charger)
            charging = sum(1 for a in self.agent_states if a.is_charging)
            pbar.set_postfix(done=done, q=f"{q:.2f}", agents=f"{active}/{self.num_agents}", nav=f"{nav_chg}/{self.num_agents}", chg=f"{charging}/{self.num_agents}", t=self.current_timestep, refresh=False)
            # Main simulation loop logic:
            # 1. Get current task assignments from GCBBA
            # 2. Update AgentState with new assignments
            # 3. Call collision avoidance for path planning/replanning
            # 4. Step simulation forward and update AgentState with new positions and task statuses
            # 5. Trigger GCBBA replanning at specified intervals or when certain conditions are met (e.g. task completion, new tasks added, rerun time etc.)
            
            if self.completed_task_ids >= self.all_task_ids and self.task_arrival_rate == 0 and self.all_task_ids:
                tqdm.write(f"All {len(self.completed_task_ids)} tasks completed at t={self.current_timestep}.")
                break
    
    def step(self) -> OrchestratorEvents:
        """
        Step the simulation forward by one timestep.
        """

        # Inject tasks 
        new_task_ids = self._inject_new_tasks()

        if self.current_timestep == 0 or self.last_gcbba_timestep < 0:
            self.run_allocation()
            self._plan_paths()

        # Track charging state before stepping so we can detect transitions
        prev_charging_busy = {
            ast.agent_id: (ast.is_charging or ast.is_navigating_to_charger)
            for ast in self.agent_states
        }

        completed_task_ids: List[int] = []
        for agent_state in self.agent_states:
            completed = agent_state.step(self.current_timestep)
            if completed and agent_state.completed_tasks:
                completed_task_ids.append(agent_state.completed_tasks[-1].task_id)

        for task_id in completed_task_ids:
            self.completed_task_ids.add(task_id)
            self._pending_task_ids.discard(task_id)

        # Energy and Charging Logic
        newly_available_agents = [
            ast.agent_id for ast in self.agent_states
            if prev_charging_busy.get(ast.agent_id) and not ast.is_charging and not ast.is_navigating_to_charger
        ]
        newly_charging_agents = self._check_and_start_charging()

        if newly_charging_agents or newly_available_agents:
            self.run_allocation()

        # New Task arrived and idle agents are available — trigger allocation immediately to assign them (instead of waiting for the next scheduled rerun)
        if new_task_ids and (self.current_timestep - self.last_gcbba_timestep) >= self.new_task_cooldown:
            if any(a.is_idle and not a.is_charging and not a.is_navigating_to_charger
                   for a in self.agent_states):
                self.run_allocation()

        self._dispatch_idle_agents_to_charge()

        self._plan_paths()

        # Check if we need to rerun GCBBA
        events = self._detect_events(completed_task_ids)

        if events.gcbba_rerun and self.last_gcbba_timestep != self.current_timestep:
            self.run_allocation()
            self._plan_paths()  # Replan paths immediately after GCBBA to reflect new assignments

        mean_depth = (
            float(np.mean(list(self._induct_queue_depth.values()))) if self._induct_queue_depth else 0
        )
        self._queue_depth_snapshots.append(mean_depth)
        
        self.current_timestep += 1
        return events

    def run_allocation(self) -> None:
        """
        Run the chosen allocation method: GCBBA, SGA, CBBA, or DMCHBA.
        """
        
        # Tasks to Exclude: completed tasks + currently executing tasks (to avoid reassigning them)
        executing_task_ids = self._get_executing_task_ids()
        excluded_task_ids = self.completed_task_ids | executing_task_ids

        # Agents to Exclude: currently charging or navigating to charger
        active_agent_indices = [
            i for i, agent_state in enumerate(self.agent_states)
            if not agent_state.is_charging and not agent_state.is_navigating_to_charger
        ]

        if len(active_agent_indices) == 0:
            remaining_tasks = len(self.all_task_ids) - len(self.completed_task_ids) - len(self._get_executing_task_ids())
            if remaining_tasks > 0:
                # All agents are charging simultaneously while work remains — throughput is halted.
                # This is recoverable: allocation re-runs automatically when agents finish charging
                # (detected via newly_available_agents in step()). Not a deadlock, but a convoy pause.
                charging_count = sum(1 for a in self.agent_states if a.is_charging or a.is_navigating_to_charger)
                min_charge_remaining = min(
                    (a.charge_remaining for a in self.agent_states if a.is_charging),
                    default=0
                )
                tqdm.write(
                    f"[t={self.current_timestep}] WARNING: All {charging_count} agents charging — "
                    f"{remaining_tasks} tasks stalled. Resuming in ~{max(1, min_charge_remaining)} timesteps."
                )
            else:
                # tqdm.write(f"[t={self.current_timestep}] No active agents. Skipping allocation.")
                pass
            self.last_gcbba_timestep = self.current_timestep
            return

        active_char_t = []
        active_task_ids = []

        for original_id in sorted(self.all_task_ids):
            if original_id not in excluded_task_ids:
                active_char_t.append(self.all_char_t[original_id])
                active_task_ids.append(original_id)
        
        if len(active_char_t) == 0:
            # Nothing to allocate - updating agent states with empty lists
            for agent_state in self.agent_states:
                agent_state.update_from_gcbba([], self.current_timestep)
            self.latest_assignment = [[] for _ in range(self.num_agents)]
            self.last_gcbba_timestep = self.current_timestep
            # tqdm.write(f"No active tasks to allocate at timestep {self.current_timestep}. Skipping GCBBA run.")
            return
        
        # Build updated char_a only for active (non-charging) agents
        updated_char_a = []
        for i in active_agent_indices:
            agent_state = self.agent_states[i]
            if self.current_timestep > 0:
                predicted_pos = agent_state.get_predicted_position(self.prediction_horizon)
                continuous_pos = self.grid_map.grid_to_continuous(*predicted_pos)
            else:
                continuous_pos = (self.all_char_a[i][0], self.all_char_a[i][1], self.all_char_a[i][2])

            speed = float(self.all_char_a[i][3])
            agent_id_value = int(self.all_char_a[i][4])
            updated_char_a.append(np.array([continuous_pos[0], continuous_pos[1], continuous_pos[2], speed, agent_id_value]))

        # Recomputing G and D based on active agent positions only
        current_positions = []
        for i in active_agent_indices:
            agent_state = self.agent_states[i]
            pos = agent_state.get_position()
            continuous_pos = self.grid_map.grid_to_continuous(*pos)
            current_positions.append((continuous_pos[0], continuous_pos[1], continuous_pos[2], agent_state.agent_id))
        
        raw_graph, G = create_graph_with_range(current_positions, self.comm_range)
        if raw_graph.number_of_nodes() == 0:
            D = 1
        else:
            if nx.is_connected(raw_graph):
                D = nx.diameter(raw_graph)
            else:
                # If the graph is not fully connected, we can take the maximum diameter of the connected components as an approximation
                D = max(nx.diameter(raw_graph.subgraph(c)) for c in nx.connected_components(raw_graph))

        # Computing Capacity
        na = len(active_agent_indices)
        nt_active = len(active_char_t)
        if self.Lt is None:
            Lt = max(3, int(np.ceil(nt_active / na)))
        else:
            Lt = self.Lt
        
        # Dispatch to the appropriate orchestrator based on the chosen allocation method
        t_allocation_start = time.perf_counter()

        if self.allocation_method == "gcbba":
            # Pass current energy levels so agents won't bid on tasks they can't afford
            agent_energies = [self.agent_states[i].energy for i in active_agent_indices]
            # Fresh GCBBA Orchestrator instance with updated parameters and state
            allocator = GCBBA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids,
                                           grid_map=self.grid_map, agent_energies=agent_energies,
                                           charging_station_grids=self.charging_station_grid_positions)
        
        elif self.allocation_method == "sga":
            agent_energies = [self.agent_states[i].energy for i in active_agent_indices]
            allocator = SGA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids,
                                         grid_map=self.grid_map, agent_energies=agent_energies,
                                         charging_station_grids=self.charging_station_grid_positions)

        elif self.allocation_method == "cbba":
            agent_energies = [self.agent_states[i].energy for i in active_agent_indices]
            allocator = CBBA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids,
                                          grid_map=self.grid_map, agent_energies=agent_energies,
                                          charging_station_grids=self.charging_station_grid_positions)

        elif self.allocation_method == "dmchba":
            agent_energies = [self.agent_states[i].energy for i in active_agent_indices]
            allocator = DMCHBA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids,
                                            grid_map=self.grid_map, agent_energies=agent_energies,
                                            charging_station_grids=self.charging_station_grid_positions)

        assignment, total_score, makespan = allocator.launch_agents(timeout_s=self.allocation_timeout_s)

        if hasattr(allocator, 'total_consensus_rounds'):
            self._last_consensus_rounds = allocator.total_consensus_rounds
            self._last_convergence_iteration = allocator.convergence_iteration
            self._total_consensus_rounds_all_calls += allocator.total_consensus_rounds
            self._total_convergence_iterations_all_calls += allocator.convergence_iteration
            self._num_allocation_calls_with_consensus += 1
        else:
            self._last_consensus_rounds = 0
            self._last_convergence_iteration = 0

        t_allocation_end = time.perf_counter()

        allocation_time_ms = (t_allocation_end - t_allocation_start) * 1000

        tqdm.write(
            f"[t={self.current_timestep}] {self.allocation_method.upper()}: "
            f"{nt_active} active tasks, "
            f"{len(excluded_task_ids)} excluded "
            f"({len(self.completed_task_ids)} done, "
            f"{len(executing_task_ids)} executing). "
            f"Score={total_score:.2f}, Makespan={makespan:.2f}, Time={allocation_time_ms:.2f}ms"
        )
        
        # If a timeout cancelled this call, bail out before mutating any state.
        # The zombie thread may still reach this point after the main thread moved on.
        if self._allocation_cancelled:
            return

        self.latest_assignment = assignment

        # _build_assignment_dict uses 0-indexed positions (allocator output); remap to original agent indices
        gcbba_assignments_by_active_pos = self._build_assignment_dict(assignment)

        for active_pos, original_idx in enumerate(active_agent_indices):
            tasks_for_agent = gcbba_assignments_by_active_pos.get(active_pos, [])
            agent_state = self.agent_states[original_idx]

            # Decrement queue depth for tasks being claimed for the first time
            for task_dict in tasks_for_agent:
                tid = task_dict['task_id']
                if tid in self._pending_task_ids:
                    self._pending_task_ids.discard(tid)
                    induct_idx = self._task_to_induct.get(tid)
                    if induct_idx is not None:
                        self._induct_queue_depth[induct_idx] = max(
                            0, self._induct_queue_depth[induct_idx] - 1
                        )

            agent_state.update_from_gcbba(tasks_for_agent, self.current_timestep)
            if agent_state.has_tasks() and agent_state.current_path is None:
                agent_state.needs_new_path = True

        # Charging/navigating agents get empty updates (preserves their charging state)
        inactive_indices = set(range(self.num_agents)) - set(active_agent_indices)
        for i in inactive_indices:
            self.agent_states[i].update_from_gcbba([], self.current_timestep)
        
        self.last_gcbba_timestep = self.current_timestep
        self._completed_at_last_gcbba = len(self.completed_task_ids)

        # Orphan detection: any task not completed and not in any agent's queue
        # should be in _pending_task_ids so Trigger 2 can pick it up.
        claimed_ids = set()
        for ast in self.agent_states:
            if ast.current_task is not None:
                claimed_ids.add(ast.current_task.task_id)
            for t in ast.planned_tasks:
                claimed_ids.add(t.task_id)
        for tid in self.all_task_ids:
            if tid not in self.completed_task_ids and tid not in claimed_ids:
                self._pending_task_ids.add(tid)

    def _get_executing_task_ids(self) -> Set[int]:
        executing_task_ids = set()
        for agent_state in self.agent_states:
            if agent_state.current_task is not None:
                executing_task_ids.add(agent_state.current_task.task_id)
        return executing_task_ids

    
    
    
    def _build_assignment_dict(self, assignment: List[List[int]]) -> Dict[int, List[int]]:
        assignments_dict: Dict[int, List[int]] = {}

        for agent_idx, task_ids in enumerate(assignment):
            tasks_for_agent: List[Dict] = []

            for task_id in task_ids:
                if task_id in self.completed_task_ids:
                    continue  # Skip already completed tasks

                char_t = self.all_char_t[task_id]

                induct_grid_pos = self.grid_map.continuous_to_grid(float(char_t[0]), float(char_t[1]), float(char_t[2]))
                eject_grid_pos = self.grid_map.continuous_to_grid(float(char_t[3]), float(char_t[4]), float(char_t[5]))

                tasks_for_agent.append({
                    "task_id": int(task_id),
                    "induct_pos": list(induct_grid_pos), # induct_grid_pos is a tuple, convert to list for easier handling in AgentState
                    "eject_pos": list(eject_grid_pos)   # eject_grid_pos
                })        
            
            assignments_dict[agent_idx] = tasks_for_agent
        
        return assignments_dict

    def _plan_paths(self) -> None:
        replan_agents = [agent_state for agent_state in self.agent_states if agent_state.needs_new_path]
        random.shuffle(replan_agents)  # Randomise order to avoid systematic priority bias

        if not replan_agents:
            return

        # Pre-planning pass (orchestration) ---------------------------------
        # Handle idle agents and apply energy-safety gates that may redirect
        # agents to a charger.  These depend on orchestrator state and must
        # run before path_planner.plan_all() is called.
        planning_agents = []
        for agent_state in replan_agents:
            goal = agent_state.get_current_goal()

            if goal is None:
                # Idle agent: hold current position so others plan around it.
                # In multi-planner mode use the task planner's table (the one task agents read).
                hold_planner = (
                    self.planner_map["task"] if self.planner_map else self.path_planner
                )
                hold_planner.hold_position(
                    agent_state.get_position(), agent_state.agent_id, self.current_timestep
                )
                agent_state.needs_new_path = False
                continue

            start = agent_state.get_position()

            # At the eject->induct transition (first path plan for a new to_induct leg),
            # verify the agent has enough energy for the full cycle: induct leg + eject leg
            # + reach charger. No buffer -- this is a hard floor. The to_eject gate below
            # provides the conservative (1.3x) safety net for congestion-induced overruns.
            if (agent_state.task_phase == "to_induct"
                    and agent_state.current_task is not None
                    and agent_state.current_path is None):
                induct_pos = agent_state.current_task.induct_pos
                eject_pos  = agent_state.current_task.eject_pos
                dist_to_induct       = self._bfs_dist(start, induct_pos)
                dist_induct_to_eject = self._bfs_dist(induct_pos, eject_pos)
                charger_dist, charger_pos = self._get_nearest_charger_from_pos(eject_pos)

                if charger_pos is not None:
                    energy_needed = dist_to_induct + dist_induct_to_eject + charger_dist
                    if agent_state.energy < energy_needed:
                        aborted_task_id = agent_state.current_task.task_id
                        tqdm.write(
                            f"[t={self.current_timestep}] Agent {agent_state.agent_id}: energy "
                            f"({agent_state.energy}) too low for full cycle -- induct ({dist_to_induct}) "
                            f"+ eject ({dist_induct_to_eject}) + charger ({charger_dist}) = {energy_needed}. "
                            f"Dropping task {aborted_task_id} before starting induct leg."
                        )
                        for t in agent_state.planned_tasks:
                            self._pending_task_ids.add(t.task_id)
                        agent_state.start_charging(charger_pos)
                        self._pending_task_ids.add(aborted_task_id)
                        induct_idx = self._task_to_induct.get(aborted_task_id)
                        if induct_idx is not None:
                            self._induct_queue_depth[induct_idx] += 1

            # At the induct->eject transition, verify the agent can complete the eject
            # leg AND reach a charger afterward.  Runs unconditionally -- even for
            # single-task agents (planned_tasks == []) -- because the to_eject skip in
            # _check_and_start_charging relies on this gate having already run.
            if (agent_state.task_phase == "to_eject"
                    and agent_state.current_task is not None):
                eject_pos = agent_state.current_task.eject_pos
                dist_to_eject = self._bfs_dist(start, eject_pos)
                charger_dist_from_eject, charger_pos_from_eject = self._get_nearest_charger_from_pos(eject_pos)

                # During to_eject, never drop the current task: payload is already picked.
                # If energy is low, continue toward eject and let charging happen afterward.
                if charger_pos_from_eject is not None:
                    energy_to_survive = dist_to_eject + charger_dist_from_eject
                    if agent_state.energy < int(energy_to_survive * 1.1):
                        tqdm.write(
                            f"[t={self.current_timestep}] Agent {agent_state.agent_id}: energy "
                            f"({agent_state.energy}) too low for eject ({dist_to_eject} steps) + "
                            f"charger ({charger_dist_from_eject} steps). "
                            f"Continuing to_eject without dropping task "
                            f"{agent_state.current_task.task_id} (payload already picked)."
                        )

            planning_agents.append(agent_state)

        # Path planning ------------------------------------------------------
        if not planning_agents:
            return

        paths = self.path_planner.plan_all(
            planning_agents,
            self.current_timestep,
            self.max_plan_time,
            planner_map=self.planner_map
        )
        for agent_state in planning_agents:
            agent_state.assign_path(paths[agent_state.agent_id])

            # If a task repeatedly fails to get a usable path, release and reassign it.
            if (
                agent_state.current_task is not None
                and agent_state.current_path is None
                and agent_state.needs_new_path
                and agent_state.task_phase != "to_eject"
                and agent_state.no_path_replans >= self.no_path_replan_limit
            ):
                abandoned_ids = agent_state.abandon_tasks_for_reallocation()
                for tid in abandoned_ids:
                    self._pending_task_ids.add(tid)
            
    def save_trajectories(self, path: str = "results/data/trajectories.csv") -> None:
        """Export all agent position histories to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "x", "y", "z", "timestep"])
            for agent_state in self.agent_states:
                for (x, y, z, t) in agent_state.position_history:
                    writer.writerow([agent_state.agent_id, x, y, z, t])
        print(f"Trajectories saved to {path}")

    def _detect_events(self, completed_task_ids: List[int]) -> OrchestratorEvents:

        stuck_agent_ids: List[int] = []
        released_task_ids: List[int] = []

        for agent_state in self.agent_states:
            is_stuck = agent_state.detect_stuck(self.stuck_threshold)
            if is_stuck:
                stuck_agent_ids.append(agent_state.agent_id)
                if agent_state.current_task is not None:
                    self._stuck_task_steps[agent_state.agent_id] = self._stuck_task_steps.get(agent_state.agent_id, 0) + 1
                    if (
                        agent_state.task_phase != "to_eject"
                        and self._stuck_task_steps[agent_state.agent_id] >= self.stuck_task_release_after
                    ):
                        released = agent_state.release_claimed_tasks()
                        released_task_ids.extend(released)
                        self._stuck_task_steps[agent_state.agent_id] = 0
                    elif agent_state.task_phase == "to_eject":
                        # Never drop a payload already picked up; keep replanning.
                        agent_state.needs_new_path = True
                else:
                    agent_state.needs_new_path = True  # Trigger replanning for stuck agents
                    self._stuck_task_steps[agent_state.agent_id] = 0
            else:
                self._stuck_task_steps[agent_state.agent_id] = 0

        for tid in released_task_ids:
            self._pending_task_ids.add(tid)

        # Per-step orphan protection: keep all uncompleted/unclaimed tasks re-queueable.
        claimed_ids = set()
        for ast in self.agent_states:
            if ast.current_task is not None:
                claimed_ids.add(ast.current_task.task_id)
            for t in ast.planned_tasks:
                claimed_ids.add(t.task_id)
        orphan_added = False
        for tid in self.all_task_ids:
            if tid in self.completed_task_ids:
                continue
            if tid in claimed_ids:
                continue
            if tid not in self._pending_task_ids:
                self._pending_task_ids.add(tid)
                orphan_added = True

        gcbba_rerun = False
        time_since_last_gcbba = self.current_timestep - self.last_gcbba_timestep

        batch_threshold = max(2, self.num_agents // 3)  # Threshold for batch triggering based on number of agents
        completed_since_last = len(self.completed_task_ids) - self._completed_at_last_gcbba
        if completed_since_last >= batch_threshold:
            gcbba_rerun = True  # Trigger rerun if enough tasks completed since last run

        # Trigger rerun when unassigned tasks are pending and idle agents are available
        if not gcbba_rerun and self._pending_task_ids and time_since_last_gcbba >= self.new_task_cooldown:
            if any(a.is_idle and not a.is_charging and not a.is_navigating_to_charger
                   for a in self.agent_states):
                gcbba_rerun = True

        if released_task_ids:
            gcbba_rerun = True

        if orphan_added or self._force_allocation_rerun:
            gcbba_rerun = True
            self._force_allocation_rerun = False

        # Rerun Interval: if nothing has triggered a rerun for rerun_interval timesteps, force one
        if not gcbba_rerun and time_since_last_gcbba >= self.rerun_interval:
            gcbba_rerun = True

        return OrchestratorEvents(
            completed_task_ids=completed_task_ids,
            stuck_agent_ids=stuck_agent_ids,
            gcbba_rerun=gcbba_rerun
        )

    
    
    
    #################### Energy and Charging Logic (Optional) ####################
    def _bfs_dist(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
        """BFS obstacle-aware distance between two grid positions.
        Uses precomputed BFS tables keyed by station position; falls back to
        Manhattan distance if neither endpoint has a precomputed table.
        """
        table = self.grid_map.bfs_distances_from_station.get(b)
        if table is not None and a in table:
            return table[a]
        table = self.grid_map.bfs_distances_from_station.get(a)
        if table is not None and b in table:
            return table[b]
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _get_nearest_charger_from_pos(self, grid_pos: Tuple[int, int, int]) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
        """
        Returns (bfs_distance, charger_grid_pos) for the nearest charging station
        from an arbitrary grid position. Returns (None, None) if unreachable.
        """
        best_dist = float('inf')
        best_pos = None
        for station_grid_pos in self.charging_station_grid_positions:
            dist_map = self.grid_map.bfs_distances_from_station.get(station_grid_pos, {})
            dist = dist_map.get(grid_pos, float('inf'))
            if dist < best_dist:
                best_dist = dist
                best_pos = station_grid_pos
        if best_dist == float('inf'):
            return None, None
        return int(best_dist), best_pos

    def _get_nearest_charging_station(self, agent_state: AgentState) -> Tuple[int, Tuple[int, int, int]]:
        """
        Returns (bfs_distance, grid_pos) for the nearest charging station to the agent.
        Uses precomputed BFS distance tables (keyed by station grid position).
        Multiple agents may share the same charging station.
        """
        agent_pos = agent_state.get_position()
        dist, pos = self._get_nearest_charger_from_pos(agent_pos)
        return (dist if dist is not None else 0), pos

    def _get_nearest_wait_from_pos(
        self,
        grid_pos: Tuple[int, int, int],
        excluded_wait_positions: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
        """
        Returns (bfs_distance, idle_task_grid_pos) for the nearest configured
        idle-task station from an arbitrary grid position.
        """
        if not self.idle_task_grid_positions:
            return None, None

        excluded = excluded_wait_positions or set()

        best_dist = float('inf')
        best_pos = None
        for wait_grid_pos in self.idle_task_grid_positions:
            if wait_grid_pos in excluded:
                continue
            if wait_grid_pos in self.wait_forbidden_positions:
                continue
            dist_map = self.grid_map.bfs_distances_from_station.get(wait_grid_pos, {})
            dist = dist_map.get(grid_pos, float('inf'))
            if dist < best_dist:
                best_dist = dist
                best_pos = wait_grid_pos
        if best_dist == float('inf'):
            return None, None
        return int(best_dist), best_pos

    def _dispatch_idle_agents_to_charge(self) -> None:
        """
        If an agent has no current/planned tasks for several timesteps, send it
        to a configured idle_task station (typically edge cells behind inducts)
        so it stays out of critical traffic while remaining allocatable.
        """
        if self.idle_wait_after <= 0:
            return

        # Keep agents available for work while there are still unclaimed tasks waiting.
        if self._pending_task_ids:
            return

        reserved_wait_positions: Set[Tuple[int, int, int]] = set()
        for agent_state in self.agent_states:
            if agent_state.is_navigating_to_wait and agent_state.wait_position is not None:
                reserved_wait_positions.add(agent_state.wait_position)
            pos = agent_state.get_position()
            if pos in self.idle_task_grid_positions:
                reserved_wait_positions.add(pos)

        dynamic_forbidden_positions: Set[Tuple[int, int, int]] = set()
        for agent_state in self.agent_states:
            if agent_state.current_task is None:
                continue
            dynamic_forbidden_positions.add(agent_state.get_position())
            goal = agent_state.get_current_goal()
            if goal is not None:
                dynamic_forbidden_positions.add(goal)

        for agent_state in self.agent_states:
            if agent_state.is_charging or agent_state.is_navigating_to_charger or agent_state.is_navigating_to_wait:
                continue
            if agent_state.current_task is not None or len(agent_state.planned_tasks) > 0:
                continue
            if not agent_state.no_current_task:
                continue
            if agent_state.get_position() in self.idle_task_grid_positions:
                continue

            _, wait_pos = self._get_nearest_wait_from_pos(
                agent_state.get_position(),
                excluded_wait_positions=reserved_wait_positions | dynamic_forbidden_positions,
            )
            if wait_pos is None:
                continue
            if agent_state.get_position() == wait_pos:
                continue

            agent_state.start_waiting(wait_pos)
            reserved_wait_positions.add(wait_pos)

    def _check_and_start_charging(self) -> List[int]:
        """
        Check if any agents need to start charging and update their state accordingly.
        Checks both active AND idle agents. Idle agents must be checked proactively to
        prevent task-thrashing: without this, a low-energy idle agent gets assigned a task,
        immediately drops it to charge, gets reassigned, and loops indefinitely.
        """
        newly_charging_agents = []

        for agent_state in self.agent_states:
            # Only skip agents already handling their charging — idle agents are checked too
            if agent_state.is_charging or agent_state.is_navigating_to_charger or agent_state.is_navigating_to_wait:
                continue

            # Never interrupt an agent mid-eject — the induct-station energy check
            # (in _plan_paths) already decided this agent has enough energy to finish.
            if agent_state.task_phase == "to_eject":
                continue

            dist, nearest_charging_station = self._get_nearest_charging_station(agent_state)

            if nearest_charging_station is None:
                continue

            if agent_state.needs_charging(dist, multiplier=self.charging_trigger_multiplier):
                # Re-queue tasks about to be dropped so _detect_events() can reassign them
                if agent_state.current_task is not None:
                    self._pending_task_ids.add(agent_state.current_task.task_id)
                for t in agent_state.planned_tasks:
                    self._pending_task_ids.add(t.task_id)
                agent_state.start_charging(nearest_charging_station, charge_duration=self.charge_duration)
                newly_charging_agents.append(agent_state.agent_id)

        return newly_charging_agents
    
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "gridworld_warehouse_small.yaml")

    orchestrator = IntegrationOrchestrator(config_path)

    t0 = time.time()
    orchestrator.run_simulation(timesteps=800)
    tf = time.time()

    print(f"Simulation completed in {tf - t0} seconds.")
    orchestrator.save_trajectories()

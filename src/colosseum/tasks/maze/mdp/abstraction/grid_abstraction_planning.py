from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from loguru import logger

from colosseum.managers.abstraction_manager import AbstractionTermCfg
from colosseum.tasks.maze.mdp.abstraction.grid_abstraction import (
    GridAbstraction,
    GridAbstractionTermCfg,
)

if TYPE_CHECKING:
    from unified_planning.plans import SequentialPlan
    from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


@dataclass(kw_only=True)
class PlannedAbstractionTermCfg(GridAbstractionTermCfg):
    """Configuration for PlannedAbstraction.

    Inherits all GridAbstractionTermCfg parameters.
    Plans are provided at initialization time.
    """

    def build(self, env: AbstractionBasedEnv) -> PlannedAbstraction:
        return PlannedAbstraction(cfg=self, env=env)


class PlannedAbstraction(GridAbstraction):
    """Abstraction based on pre-computed planner solutions.

    Instead of computing the cost map via Dijkstra/BFS, this class uses
    planned action sequences to guide the robot. Useful when you have:

    1. Pre-solved all maze configurations offline using a planner
    2. Want to use optimal/near-optimal solutions as oracle guidance

    The plan is converted into a cost map where cells visited earlier
    have lower cost, creating a gradient towards the goal.

    Usage:
        plans = [plan_for_env_0, plan_for_env_1, ...]
        abstraction.set_plans(plans)
    """

    cfg: PlannedAbstractionTermCfg

    def __init__(self, cfg: PlannedAbstractionTermCfg, env: AbstractionBasedEnv):
        super().__init__(cfg, env)
        self.plans: list[Optional[SequentialPlan]] = [None] * self.num_envs
        self.plan_step_indices: dict[int, dict[tuple, int]] = {}
        logger.info("Initialized PlannedAbstraction (plans not yet set)")

    def set_plans(self, plans: list[SequentialPlan]) -> None:
        """Set pre-computed plans for each environment.

        Args:
            plans: List of SequentialPlan objects, one per environment.
                   Index i corresponds to environment i.
        """
        if len(plans) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} plans, got {len(plans)}"
            )

        self.plans = plans

        # Parse plans into location sequences for fast lookup
        self.plan_step_indices = {}
        for env_idx, plan in enumerate(plans):
            if plan is None:
                logger.warning(f"Environment {env_idx} has no plan")
                continue

            step_dict = {}
            for step_idx, action in enumerate(plan.actions):
                # Extract locations from action parameters
                action_params = action.parameters
                if len(action_params) > 0:
                    locations = tuple(str(p) for p in action_params)
                    step_dict[locations] = step_idx

            self.plan_step_indices[env_idx] = step_dict
            logger.info(
                f"Environment {env_idx}: parsed {len(step_dict)} steps from plan"
            )

    def _compute_costs_parallel(self) -> torch.Tensor:
        """Compute cost map from pre-computed plans.

        Cost is based on the step index in the plan:
        - Earlier steps (closer to goal) have lower cost
        - Later steps have higher cost
        - Unvisited cells have high cost (not on the plan path)

        Returns:
            [num_envs, rows, cols] cost tensor
        """
        assert self.settings is not None
        rows = self.grid_frame.num_rows
        cols = self.grid_frame.num_cols

        cost_map = torch.full(
            (self.num_envs, rows, cols),
            self.OBSTACLE_COST,
            dtype=torch.float32,
            device=self.device,
        )

        goal_indices = self._local_to_grid(self.settings.goal)

        for env_idx in range(self.num_envs):
            if self.plans[env_idx] is None:
                logger.warning(f"No plan for environment {env_idx}, using obstacles only")
                continue

            # Mark obstacles
            cost_map[env_idx, self.obstacle_mask] = self.OBSTACLE_COST

            # Set goal cell
            gi, gj = int(goal_indices[env_idx, 0].item()), int(goal_indices[env_idx, 1].item())
            if 0 <= gi < rows and 0 <= gj < cols:
                cost_map[env_idx, gi, gj] = self.GOAL_COST

            # Assign costs based on plan steps
            if env_idx not in self.plan_step_indices:
                logger.warning(f"Environment {env_idx} has unparsed plan")
                continue

            step_dict = self.plan_step_indices[env_idx]
            plan = self.plans[env_idx]

            # Mark all cells visited in the plan
            # Cost = max_step - step_index (so goal step has low cost, early steps high)
            max_step = len(plan.actions)

            for action in plan.actions:
                action_params = action.parameters
                if len(action_params) == 0:
                    continue

                # For move actions: player location (first param) is visited
                # For push_box actions: player ends at second param after push
                if action.name == "move":
                    target_loc_str = str(action_params[1])
                elif action.name == "push-box":
                    # After push-box, player is at middle location (y in x,y,z)
                    target_loc_str = str(action_params[1])
                else:
                    continue

                # Convert location string to grid indices
                grid_idx = self._location_str_to_grid_idx(target_loc_str)
                if grid_idx is None:
                    continue

                i, j = grid_idx
                if not (0 <= i < rows and 0 <= j < cols):
                    continue

                # Cells on plan path have cost proportional to steps remaining
                locations_key = tuple(str(p) for p in action_params)
                if locations_key in step_dict:
                    step_idx = step_dict[locations_key]
                    cost = (max_step - step_idx) / max_step
                    cost_map[env_idx, i, j] = cost

        return cost_map

    def _location_str_to_grid_idx(self, loc_str: str) -> Optional[tuple[int, int]]:
        """Convert location string like 'loc-3-5' to grid indices (i, j).

        Args:
            loc_str: Location identifier, e.g., 'loc-3-5'

        Returns:
            (i, j) grid indices, or None if parsing fails
        """
        try:
            parts = loc_str.split("-")
            if len(parts) >= 3 and parts[0] == "loc":
                x = int(parts[1])
                y = int(parts[2])
                return (y, x)  # Note: grid uses (row, col) = (y, x)
        except (ValueError, IndexError):
            pass
        return None

    def _compute_direction_map_gradient(self, cost_map: torch.Tensor) -> torch.Tensor:
        """Direction field from plan-based cost map using gradient descent.

        For plan-based costs, gradient descent works well since we explicitly
        assigned costs based on plan steps.
        """
        return super()._compute_direction_map_gradient(cost_map)

    def _compute_direction_map_harmonic(self, cost_map: torch.Tensor) -> torch.Tensor:
        """Smooth the plan-based cost map before direction field generation.

        This softens any sharp transitions in the plan-based cost field.
        """
        return super()._compute_direction_map_harmonic(cost_map)


# Helpers

def parse_plan_to_trajectory(plan: SequentialPlan) -> list[str]:
    """Extract visited locations from a plan in order.

    Args:
        plan: SequentialPlan from unified_planning

    Returns:
        List of location strings visited by the plan
    """
    trajectory = []
    for action in plan.actions:
        action_params = action.parameters

        if action.name == "move" and len(action_params) >= 2:
            trajectory.append(str(action_params[1]))
        elif action.name == "push-box" and len(action_params) >= 2:
            trajectory.append(str(action_params[1]))

    return trajectory


def solve_all_environments_offline(
        env_ids: list[int],
        level_generator_fn,
        planner_fn,
        with_axioms: bool = False,
) -> dict[int, Optional[SequentialPlan]]:
    """Pre-compute plans for all training environments.

    Call this at task initialization to generate and solve all mazes.

    Args:
        env_ids: List of environment IDs to solve
        level_generator_fn: Function that takes env_id and returns level string
        planner_fn: Function that takes (level_string, with_axioms) and returns plan
        with_axioms: Whether to use axioms in planning

    Returns:
        Dict mapping env_id -> SequentialPlan (or None if unsolvable)

    Example:
        def level_gen(env_id):
            maze = generate_maze(20, 20, seed=env_id)
            return maze_to_string(maze)

        def plan_gen(level_str, with_axioms):
            problem = create_sokoban_problem(level_str, with_axioms=with_axioms)
            with OneshotPlanner(...) as planner:
                result = planner.solve(problem)
                return result.plan if result.status in POSITIVE_OUTCOMES else None

        plans = solve_all_environments_offline(
            env_ids=list(range(64)),
            level_generator_fn=level_gen,
            planner_fn=plan_gen,
            with_axioms=False,
        )
    """
    plans = {}

    for env_id in env_ids:
        try:
            level_str = level_generator_fn(env_id)
            plan = planner_fn(level_str, with_axioms)

            if plan is not None:
                logger.info(f"Environment {env_id}: found plan with {len(plan.actions)} actions")
            else:
                logger.warning(f"Environment {env_id}: no plan found")

            plans[env_id] = plan
        except Exception as e:
            logger.error(f"Environment {env_id}: planning failed: {e}")
            plans[env_id] = None

    solved = sum(1 for p in plans.values() if p is not None)
    logger.info(f"Solved {solved}/{len(env_ids)} environments offline")

    return plans


# TODO remove this after we add the task description
def create_planned_abstraction_cfg(
        maze,
        resolution_factor: int = 3,
        wall_center_weight: float = 2.0,
) -> dict[str, AbstractionTermCfg]:
    """Helper to create config dict for task file.

    Usage in your task config:
        abstractions=create_planned_abstraction_cfg(maze)
    """
    grid_frame = maze.build_upsampled_grid_frame(resolution_factor)
    obstacle_mask = maze.build_obstacle_mask(resolution_factor)

    return {
        "grid": PlannedAbstractionTermCfg(
            grid_frame=grid_frame,
            obstacle_mask=obstacle_mask,
            direction_method="harmonic",
            wall_center_weight=wall_center_weight,
        ),
    }

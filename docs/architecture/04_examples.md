# 4. Worked examples

For every modification recipe in `03_modify_each_part.md`, this file shows
either a fully-written-out new example or walks through the existing code
that already implements it. References stay close to repo paths so you can
cross-check against the source.

---

## 4.1 Example: walking through the **velocity task** (canonical task)

The smallest task in the repo is `t1-velocity`. It uses **only** mjlab's
core managers — no Colosseum-specific extras — so it's the cleanest
illustration of how a task is wired together.

Files (under `src/colosseum/tasks/velocity/`):

```
__init__.py                        ← triggers config import
config/__init__.py
config/t1_23dof/__init__.py
config/t1_23dof/t1_velocity_cfg.py     ← env factory + @register_task
config/t1_23dof/algo_cfg.py            ← booster_t1_ppo_cfg()
config/t1_23dof/observation_cfg.py     ← actor_terms / critic_terms
config/t1_23dof/reward_cfg.py          ← rewards dict
config/t1_23dof/event_cfg.py           ← events dict
config/t1_23dof/cact_cfg.py            ← commands / actions / curriculum / terminations
```

The env factory in `t1_velocity_cfg.py`:

```python
def booster_t1_velocity_env_cfg(play=False):
    cfg = ManagerBasedRlEnvCfg(           # <- plain mjlab cfg, no extras
        scene=scene_cfg(play),            # SceneCfg with terrain + T1 + sensors
        observations=observations,        # from observation_cfg
        actions=actions,                  # from cact_cfg
        commands=commands,                # twist UniformVelocityCommand
        events=events,                    # from event_cfg
        rewards=rewards,                  # from reward_cfg
        terminations=terminations,        # time_out, fell_over
        curriculum=curriculum,            # commands_vel
        ...
    )
    if play: ...                          # play-mode tweaks
    return cfg
```

The task subclass:

```python
@register_task("t1-velocity")
@dataclass(frozen=True)
class T1VelocityTask(TaskConfig):
    name: str = "t1-velocity"
    env: ManagerBasedRlEnvCfg = field(default_factory=booster_t1_velocity_env_cfg)
    @property
    def train_env_cfg(self): return self.env
    @property
    def play_env_cfg(self):  return booster_t1_velocity_env_cfg(play=True)
    @property
    def algo_cfg(self):      return booster_t1_ppo_cfg()
```

Reading the four `*_cfg.py` files shows the canonical content of each
manager dict — copy them when you scaffold a new task.

---

## 4.2 Example: add a new task — `t1-balance` (writing it from scratch)

A minimal stand-in-place task: keep T1 upright while perturbed. We reuse
T1's sensors, the velocity-task command structure (with zero target), and
flat-orientation rewards.

### 4.2.1 Files

```
src/colosseum/tasks/balance/
  __init__.py
  config/__init__.py
  config/t1_23dof/__init__.py
  config/t1_23dof/t1_balance_cfg.py
  config/t1_23dof/algo_cfg.py
  config/t1_23dof/cact_cfg.py
  config/t1_23dof/observation_cfg.py
  config/t1_23dof/reward_cfg.py
  config/t1_23dof/event_cfg.py
```

### 4.2.2 `__init__.py` chain

```python
# src/colosseum/tasks/balance/__init__.py
try:
  import colosseum.tasks.balance.config.t1_23dof.t1_balance_cfg  # noqa: F401
except ImportError:
  pass
```

```python
# src/colosseum/tasks/balance/config/__init__.py
# (intentionally empty)
```

```python
# src/colosseum/tasks/balance/config/t1_23dof/__init__.py
from .t1_balance_cfg import *  # noqa: F401
```

### 4.2.3 `t1_balance_cfg.py`

```python
from dataclasses import dataclass, field
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
    FEET_GROUND_CONTACT_SENSOR, NONFOOT_GROUND_CONTACT_SENSOR,
    SELF_COLLISION_SENSOR,
)

from .algo_cfg import t1_balance_ppo_cfg
from .cact_cfg import actions, commands, terminations
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play=False) -> SceneCfg:
    return SceneCfg(
        terrain=TerrainEntityCfg(),
        entities={"robot": get_robot_cfg()},
        sensors=(
            FEET_GROUND_CONTACT_SENSOR,
            NONFOOT_GROUND_CONTACT_SENSOR,
            SELF_COLLISION_SENSOR,
        ),
        num_envs=1,
        extent=10.0,
    )


def viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=BASE_BODY_NAME,
        distance=3.0, elevation=-5.0, azimuth=90.0,
    )


def sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        nconmax=45, njmax=1500, contact_sensor_maxmatch=500,
        mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )


def t1_balance_env_cfg(play=False) -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(
        scene=scene_cfg(play),
        observations=observations, actions=actions, commands=commands,
        events=events, rewards=rewards, terminations=terminations,
        curriculum={}, metrics={},
        viewer=viewer_cfg(), sim=sim_cfg(),
        decimation=4, episode_length_s=20.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


@register_task("t1-balance")
@dataclass(frozen=True)
class T1BalanceTask(TaskConfig):
    name: str = "t1-balance"
    env: ManagerBasedRlEnvCfg = field(default_factory=t1_balance_env_cfg)

    @property
    def train_env_cfg(self): return self.env
    @property
    def play_env_cfg(self):  return t1_balance_env_cfg(play=True)
    @property
    def algo_cfg(self):      return t1_balance_ppo_cfg()
```

### 4.2.4 `algo_cfg.py`, `cact_cfg.py`, etc.

```python
# algo_cfg.py
from colosseum.config.types.algorithm import PpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig

def t1_balance_ppo_cfg() -> PpoConfig:
    return PpoConfig(
        name="PPO", target="colosseum.algorithm.ppo:PPO",
        learning_steps=200_000_000, num_steps_per_env=24,
        actor=PpoActorConfig(hidden_layers=[256, 256], activation="elu"),
        critic=PpoCriticConfig(hidden_layers=[256, 256], activation="elu"),
    )
```

```python
# cact_cfg.py — minimal: no command, just zero-velocity tracking
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
import math

from colosseum.robots.t1_23dof.constants import ACTION_SCALE

actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",),
        scale=ACTION_SCALE, use_default_offset=True),
}
commands = {}
terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=bad_orientation,
                                    params={"limit_angle": math.radians(60.0)}),
}
```

`observation_cfg.py`, `reward_cfg.py`, and `event_cfg.py` would be cut-down
copies of the velocity-task versions, dropping every reward that references
the (now-empty) `twist` command.

### 4.2.5 Smoke run

```
pixi run train task:t1-balance --task.env.scene.num-envs 64 \
    --task.algo-cfg.learning-steps 24000 --logger disabled
```

You should see `episode/Episode_Reward/upright` start increasing within a
few iterations.

---

## 4.3 Example: add an algorithm — minimal SAC stub

This shows how the registry plugs in. We are not implementing a full SAC
here, but the skeleton is what every new algorithm looks like.

```python
# src/colosseum/algorithm/sac.py
from typing import Callable
import torch
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.base_algorithm import BaseAlgorithm
from colosseum.config.types.algorithm import AlgorithmConfig, register_algorithm

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SacConfig(AlgorithmConfig):
    name: str = "SAC"
    target: str = "colosseum.algorithm.sac:SAC"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    gamma: float = 0.99


@register_algorithm("sac", config_class=SacConfig)
class SAC(BaseAlgorithm):
    def __init__(self, config: SacConfig, env: ManagerBasedRlEnv,
                 device, log_fn: Callable, log_interval: int):
        super().__init__(config=config, env=env, device=device,
                         log_fn=log_fn, log_interval=log_interval)
        # … build actor / critic / replay buffer …

    def train(self):
        ...

    def save(self, path, **extra): ...
    def load(self, path): ...
```

Then a task can opt into it:

```python
# in tasks/<task>/config/<robot>/algo_cfg.py
from colosseum.algorithm.sac import SacConfig

def my_task_sac_cfg() -> SacConfig:
    return SacConfig(learning_steps=10_000_000, batch_size=512)
```

```python
# inside MyTask
@property
def algo_cfg(self): return my_task_sac_cfg()
```

`scripts/train.py` uses `algo_cfg.target` to import and instantiate `SAC`,
so no changes to the script are needed.

---

## 4.4 Example: add a reward term — `feet_distance_penalty`

This term *exists* (see `src/colosseum/mdp/rewards.py::feet_distance_penalty`)
and is wired in by the **dribbling** task as a CaT constraint *and* as a
reward in earlier iterations of the velocity recipe.

Reading `mdp/rewards.py`:

```python
def feet_distance_penalty(env, asset_cfg, min_dist=0.2):
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]   # (N, 2, 3)
    base_pos_w = asset.data.root_link_pos_w[:, :3].unsqueeze(1)     # (N, 1, 3)
    quat_w = asset.data.root_link_quat_w
    quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
    left_b  = quat_apply(quat_conj, foot_pos_w[:, 0] - base_pos_w[:, 0])
    right_b = quat_apply(quat_conj, foot_pos_w[:, 1] - base_pos_w[:, 0])
    dist = (left_b[:, 1] - right_b[:, 1]).abs()
    return (min_dist - dist).clamp(min=0.0, max=min_dist)
```

To register as a reward:

```python
# in any task's reward_cfg.py
from colosseum.mdp.rewards import feet_distance_penalty
rewards["feet_distance"] = RewardTermCfg(
    func=feet_distance_penalty, weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
            "min_dist": 0.05},
)
```

To register as a CaT constraint instead (as the dribbling task does):

```python
# in tasks/dribbling/config/t1_23dof/constraint_cfg.py
dribbling_constraints["feet_distance"] = ConstraintTermCfg(
    func=feet_distance_penalty, max_p=0.5,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
            "min_dist": 0.05},
)
```

Same function, two different managers — the only thing that differs is the
config dataclass that wraps it.

---

## 4.5 Example: add an observation group — `privileged_ball`

From `tasks/dribbling/config/t1_23dof/observation_cfg.py`:

```python
privileged_ball_terms = {
    "ball_pos":    ObservationTermCfg(func=ball_position),    # (N, 2)
    "ball_vel_xy": ObservationTermCfg(func=ball_velocity_xy), # (N, 2)
}

observations["privileged_ball"] = ObservationGroupCfg(
    terms=privileged_ball_terms,
    concatenate_terms=True,
    enable_corruption=False,        # GT, no noise
)
```

The group is then **consumed** in three places:

1. The critic, which gets `**privileged_ball_terms` for asymmetric
   actor-critic training.
2. The RMA encoder term `DribblingRmaTermCfg(privileged_obs_group="privileged_ball", …)`.
3. *Nothing else* — the actor never sees this group, so the policy can be
   deployed with sensor-only observations after Phase 2.

This pattern (one group, three consumers) is the entire point of having
named groups.

---

## 4.6 Example: add an event — `push_robot`

From `tasks/velocity/config/t1_23dof/event_cfg.py`:

```python
events["push_robot"] = EventTermCfg(
    func=push_by_setting_velocity,                         # mjlab built-in
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
    }},
)
```

In `play` mode, the task factory simply pops it:

```python
if play: cfg.events.pop("push_robot", None)
```

---

## 4.7 Example: add a termination — `arrived_at_goal`

From `tasks/maze/mdp/terminations.py`:

```python
def arrived_at_goal(env, asset_cfg=SceneEntityCfg("robot"), threshold=0.5):
    pos = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids][:, 0, :2]
    goal = env.command_manager.get_command("goal")
    return torch.norm(pos - goal, dim=1) < threshold
```

Registered in `tasks/maze/config/t1_23dof/cact_cfg.py`:

```python
terminations["arrived_at_goal"] = TerminationTermCfg(
    func=arrived_at_goal,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=("root_site",)),
            "threshold": 0.5},
)
```

This name is special: `BaseAlgorithm.update_episode_counts` recognises it
and uses it for rolling success-rate tracking (the value reported in W&B
panels). Use the same key for any task with a clear "success" notion.

---

## 4.8 Example: add a curriculum — `commands_vel`

From `tasks/velocity/config/t1_23dof/cact_cfg.py`:

```python
curriculum["command_vel"] = CurriculumTermCfg(
    func=commands_vel,                    # mjlab built-in
    params={"command_name": "twist", "velocity_stages": [
        {"step": 0,         "lin_vel_x": (-1.0, 1.0), ...},
        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), ...},
        {"step": 10000 * 24,"lin_vel_x": (-2.0, 3.0), ...},
    ]},
)
```

Each stage has a `step` threshold; when `env.common_step_counter` crosses
it, `commands_vel` writes the new range into
`command_manager.get_term("twist").cfg.ranges`. Because the term reads
`cfg.ranges` on every resample, the change takes effect on the next reset
of each env.

A more involved example — `wall_collision_termination_curriculum` in
`tasks/maze/mdp/curriculums.py` — linearly ramps the *probability* of a
termination term firing, useful when you want late-training penalties
without destabilising the early policy.

---

## 4.9 Example: add a command — `BallVelocityCommand`

Live source: `src/colosseum/tasks/dribbling/mdp/ball_velocity_command.py`.
Skeleton:

```python
class BallVelocityCommand(CommandTerm):
    cfg: BallVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
        self.target_position  = torch.zeros((env.num_envs, 2), device=env.device)

    @property
    def command(self) -> torch.Tensor:
        return self.velocity_command            # (N, 3) world-frame v_xy + 0

    def _resample_command(self, env_ids):
        # sample a persistent world-frame target relative to current ball pos
        ...

    def _update_command(self):
        # recompute self.velocity_command from current ball_pos and target_pos
        ...
```

Registered in `tasks/dribbling/config/t1_23dof/cact_cfg.py`:

```python
commands["ball_vel"] = BallVelocityCommandCfg(
    target_distance_range=(1.5, 5.0),
    speed_gain=0.5, min_speed=0.1, max_speed=1.0,
    resampling_time_range=(4.0, 8.0),
)
```

Every reward / observation / curriculum that needs the ball velocity
references it via `command_name="ball_vel"`.

---

## 4.10 Example: add an RMA encoder — the `DribblingRmaTerm`

The full implementation lives at
`src/colosseum/research/dribbling/rma_terms.py` (300+ lines). The wiring
into the task is a single block in
`tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py`:

```python
cfg = ColosseumEnvCfg(
    ...
    encoders={
        "dribbling": DribblingRmaTermCfg(
            privileged_obs_group="privileged_ball",
            obstacle_privileged_obs_group="privileged_obstacles",
            adaptation_obs_group="depth_frames" if use_depth_camera else None,
        ),
    },
    use_depth_camera=use_depth_camera,
    ...
)
```

What that single config triggers, end-to-end:

1. `ColosseumEnv.load_managers` builds an `RmaManager` whose only term is
   `DribblingRmaTerm`.
2. `RmaPPO._build_networks` widens the actor input by 64 (the term's
   `latent_dim`).
3. `RmaPPO._build_optimizers` adds the privileged encoder parameters.
4. `RmaPPO._collect_rollout` stores `privileged_ball` and
   `privileged_obstacles` in the rollout buffer (because the term's
   `privileged_group_names` returns those names).
5. During Phase 1 `_learning_step` re-encodes those groups so encoder
   gradients flow.
6. `train_phase2.py` flips `algo._phase = 2`, calls
   `build_adaptation_optimizer()`, and `_train_phase2` runs MSE regression
   against the frozen Phase 1 latent. The depth encoder + its auxiliary
   ball/obstacle heads ride along.

If you want a **smaller** encoder example, follow this skeleton:

```python
@dataclass(kw_only=True)
class SimpleRmaTermCfg(RmaTermCfg):
    privileged_obs_group: str = "privileged_my"
    adaptation_obs_group: str | None = None
    latent_dim: int = 16
    def build(self, env): return SimpleRmaTerm(self, env)


class SimpleRmaTerm(RmaTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        in_dim = env.observation_manager.group_obs_dim[cfg.privileged_obs_group][0]
        self._enc = PrivilegedEncoder(in_dim, cfg.latent_dim).to(env.device)

    @property
    def privileged_encoder(self): return self._enc

    def encode_privileged(self, obs):
        return self._enc(obs[self.cfg.privileged_obs_group])
```

Drop it into a task with `cfg.encoders = {"x": SimpleRmaTermCfg()}` and
make `algo_cfg` return an `RmaPPOConfig`.

---

## 4.11 Example: add an abstraction — the maze grid

Live source: `src/colosseum/mdp/abstraction/maze/grid_abstraction.py` and
`sokoban_grid_abstraction.py`.

The grid abstraction:

1. Reads the static maze layout (an obstacle mask tensor) and the per-env
   goal cell at reset time.
2. Computes a harmonic potential field by Laplace iteration (or a Dijkstra
   cost map) once per unique `(goal_cell, map_id)` setting.
3. Exposes `get_direction(env_ids, pos_local)` returning a unit vector
   pointing along the gradient — used by `AbstractionVelocityCommand` to
   produce body-frame target velocities.

Wiring in `tasks/maze/config/t1_23dof/t1_maze_cfg.py`:

```python
def abstractions_cfg(maze, resolution_factor=3, wall_center_weight=2.0):
    return {
        "grid": GridAbstractionTermCfg(
            grid_frame=maze.build_upsampled_grid_frame(resolution_factor),
            map=maze.build_obstacle_mask(resolution_factor),
            direction_method="harmonic",
            wall_center_weight=wall_center_weight,
        ),
    }

cfg = ColosseumEnvCfg(..., abstractions=abstractions_cfg(maze))
```

The Sokoban variant in `soccer_maze` extends this idea: instead of a
per-cell direction field, it solves a multi-step block-pushing problem
with `up-symk` (a PDDL planner installed via `pixi`) and exposes the next
sub-goal cell. The pattern is the same — change-detected
`AbstractionSettings`, lazy rebuild, fast lookup at every step.

---

## 4.12 Example: add a CaT constraint — `joint_torque`

Live source: `src/colosseum/mdp/constraints.py`:

```python
def joint_torque(env, limit, asset_cfg):
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.qfrc_actuator[:, asset_cfg.joint_ids]) - limit
```

Registered in (e.g.) `tasks/dribbling/config/t1_23dof/constraint_cfg.py`:

```python
dribbling_constraints["joint_torque"] = ConstraintTermCfg(
    func=joint_torque, max_p=1.0,
    params={"limit": 40.0, "asset_cfg": SceneEntityCfg("robot")},
)
```

Then `cfg.constraints = dribbling_constraints` in the env factory.
Constraint contributions are visible in W&B as
`Episode_Constraint_violation/joint_torque` (mean % of steps violated) and
`Episode_Constraint_probability/joint_torque` (mean termination probability).

---

## 4.13 Example: add a robot — extend T1 to a hypothetical `t1_armless`

A "remove arms" robot variant does **not** need a new XML if you just want
to constrain the existing 23 DOF model. Recipe:

1. Make a sibling folder: `src/colosseum/robots/t1_armless/`.
2. Copy `constants.py`, `actuators.py`, `collisions.py`, `sensors.py` from
   `t1_23dof/`.
3. Edit `JOINT_NAMES`, `HOME_QPOS`, `ACTION_SCALE`, `ARTICULATION` to drop
   the arm joints (Shoulder, Elbow), or to set their gains so high they're
   effectively rigid.
4. Edit `get_robot_cfg(...)` to use the same XML but a different actuator
   tuple.
5. Tasks can now `from colosseum.robots.t1_armless.constants import …`.
6. (Optional) add a `tasks/<task>/config/t1_armless/` variant that swaps in
   the new robot for the existing tasks; per-robot reward stds and PD
   gains usually need re-tuning.

A genuinely different robot would also drop a new `xmls/<robot>.xml` and
mesh files into the robot folder.

---

## 4.14 Example: add a scene asset — the soccer ball

Live source: `src/colosseum/assets/ball/ball_spec.py`. It exposes:

- `BALL_MASS`, `BALL_FRICTION` (constants for observation params).
- `def get_ball_cfg() -> EntityCfg` returning a free-floating ball entity.

Plugged into the dribbling scene:

```python
return SceneCfg(
    terrain=TerrainEntityCfg(),
    sensors=tuple(sensors),
    env_spacing=10.0,
    entities={
        "ball": get_ball_cfg(),
        "robot": get_robot_cfg(foot_self_collision=True, with_head_camera=True),
        **{f"obstacle_{k}": get_obstacle_cfg(k) for k in range(NUM_OBSTACLES)},
    },
    num_envs=1,
)
```

Reset event in `tasks/dribbling/config/t1_23dof/event_cfg.py`:

```python
events["reset_ball"] = EventTermCfg(
    func=reset_root_state_uniform, mode="reset",
    params={"asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (0.5, 1.0), "y": (-0.2, 0.2), "z": (0.1, 0.1)},
            "velocity_range": {}},
)
```

Observations (`tasks/dribbling/mdp/observations.py`):

```python
def ball_position(env): return env.scene["ball"].data.root_link_pos_w[:, :2]
def ball_velocity_xy(env): return env.scene["ball"].data.root_link_lin_vel_w[:, :2]
```

These three pieces — entity factory, reset event, observation function —
are the universal "add a movable asset" template.

---

## 4.15 Example: a full multi-stage curriculum — dribbling pipeline

Live source: `src/colosseum/research/dribbling/scripts/pipeline_dribbling.py`.

```
STAGES = [
    {id: 0, name: "s0_locomotion",       p1_steps: 50M, obstacle_stage_index: 0,
     warm_start_from: None, ball_spawn_x_range: (1.5, 3.0), skip_phase2: True},
    {id: 1, name: "s1_dribbling",        p1_steps: 150M, obstacle_stage_index: 0,
     warm_start_from: "s0_locomotion_p1"},
    {id: 2, name: "s2_static_obstacle",  p1_steps: 300M, obstacle_stage_index: 1,
     warm_start_from: "s1_dribbling_p1"},
    {id: 3, name: "s3_attack_blocker",   p1_steps: 300M, obstacle_stage_index: 3,
     warm_start_from: "s2_static_obstacle_p1"},
]
```

For each stage, the pipeline:

1. Calls `pixi run train task:t1-dribbling --task.obstacle-stage-index N
   --warm-start <prev>.pt`.
2. Symlinks the resulting `latest.pt` into `<log_dir>/checkpoints/sN_p1.pt`.
3. Calls `pixi run train-phase2 --checkpoint sN_p1.pt`.
4. Symlinks Phase 2 result into `sN_p2.pt`.

This shows how curriculum + algorithm + script + model registry compose:
each stage is reproducible, swappable, and resumable.

---

## 4.16 Example: export and serve an ONNX policy

After training:

```
pixi run export-onnx task:t1-velocity --name v2 --run-name <your_run>
```

`scripts/export_onnx.py`:

1. Resolves the checkpoint via `utils.checkpoint.resolve_checkpoint`.
2. Builds `algo` (using `algo_cfg.target` to pick the class).
3. `algo.load(ckpt)` then `algo.export_onnx(out)` — the latter is
   implemented in `BaseAlgorithm.export_onnx` and traces the actor with a
   dummy input matching the proprio + (optional) latent layout.
4. `ModelRegistry.register("t1-velocity", "v2", file=…, run=…, step=…)`
   updates `models/registry.yaml` and the `default.onnx` symlink.

Then to run it in a viewer without Python-side weights:

```
pixi run play task:t1-velocity --agent onnx --onnx v2
```

`play.py::create_agent` calls `_resolve_onnx`, opens an
`onnxruntime.InferenceSession`, and wraps it as an agent callable that
takes a batched obs tensor.

---

## 4.17 Putting it all together

The `t1-dribbling` task uses **every** part of the architecture in one task:

| Architectural piece | Used by `t1-dribbling` via |
|---|---|
| Custom env class | `ColosseumEnvCfg(...)` (encoders + constraints non-empty) |
| Custom command | `BallVelocityCommand`, `GaitPhaseCommand`, `ObstacleCommand` |
| Custom asset | `assets/ball/`, `tasks/dribbling/obstacle_spec.py` |
| Task-local rewards | `tasks/dribbling/mdp/rewards.py` (`ball_vel_tracking_relaxed`, …) |
| RMA encoder | `DribblingRmaTermCfg` + Phase 1 / Phase 2 (research/dribbling/) |
| CaT constraints | `dribbling_constraints` |
| Curriculum | `curriculum["obstacle"]` (stage progression), yaw curriculum, etc. |
| Algorithm | `RmaPPO` (default) and `DaggerRmaPPO` (`--task.use-dagger`) |
| Visualisation callback | `viz_callbacks=[("camera_ball", DribblingViz)]` |
| Multi-stage pipeline | `scripts/pipeline_dribbling.py` + warm-start/ckpt symlinks |
| Phase 2 training | `scripts/train_phase2.py` |
| Evaluation protocol | `scripts/evaluate_dribbling.py` |
| ONNX export | `scripts/export_onnx.py` + `models/registry.yaml` |
| Deployment | `utils/deploy/` |

Reading those files in order is the most efficient way to internalise the
architecture once the high-level picture from `01_components.md` is clear.

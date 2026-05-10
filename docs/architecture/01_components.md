# 1. Architecture Components

Colosseum is layered on top of **mjlab**, which itself follows a three-layer
GPU-RL architecture. The Colosseum codebase reuses mjlab's simulation/scene
layers verbatim and extends mjlab's task layer with three extra managers, a
custom PPO family of algorithms, a per-task config/registry system, and
training/play/export scripts.

This document describes every component, starting from the bottom (mjlab) and
ending at the top (entry-point CLI scripts). Every component is referenced by
its file path, so you can open it directly.

---

## 1.1 The two-platform stack

```
┌──────────────────────────────────────────────────────────────────────┐
│  COLOSSEUM (this repo)                                                │
│                                                                       │
│  scripts/{train,play,train_rsl_rl,export_onnx,model_registry_cli,…}   │
│  algorithm/{ppo, rma_ppo, dagger_ppo, dagger_rma_ppo, base_algorithm} │
│  envs/colosseum_env.py        (ColosseumEnv, the unified env class)   │
│  managers/{rma, abstraction, constraint}_manager.py                   │
│  mdp/{observations, rewards, constraints, ball_rewards}.py            │
│  mdp/abstraction/maze/*  (planners, grid abstractions, soccer maze)   │
│  tasks/<task>/{config, mdp, …}    (per-task declarative configs)      │
│  robots/t1_23dof/*  +  assets/ball/*                                  │
│  config/{types, values}/*    (TaskConfig, AlgorithmConfig, registries) │
│  utils/{checkpoint, model_registry, logger, isaaclab/*, torch}        │
│  research/{dribbling, soccer_maze}/*  (papers, pipelines, evals)      │
└──────────────────────────────────────────────────────────────────────┘
                              │ extends
┌──────────────────────────────▼───────────────────────────────────────┐
│  MJLAB (external, installed via pixi/git)                             │
│                                                                       │
│  Task layer:   ManagerBasedRlEnv, ManagerBase, *Manager (Action,      │
│                Observation, Reward, Termination, Event, Command,      │
│                Curriculum), *TermCfg dataclasses                      │
│  Scene layer:  Scene, Entity, EntityIndexing, sensors, terrains       │
│  Sim layer:    Simulation (MuJoCo Warp wrapper), batched mjModel/Data │
└──────────────────────────────────────────────────────────────────────┘
```

The boundary is sharp: Colosseum never modifies mjlab classes; it subclasses
or composes them. The single new env subclass is `ColosseumEnv`
(`src/colosseum/envs/colosseum_env.py`), which inherits from
`mjlab.envs.ManagerBasedRlEnv` and conditionally instantiates extra managers.

---

## 1.2 The three layers (inherited from mjlab)

These are documented at length under `docs/mjlab/` (reachable in the
existing project docs). A short summary:

1. **Simulation Layer** — GPU-batched MuJoCo Warp physics. A single
   `Simulation` object steps thousands of envs at once and exposes
   `mjModel` / `mjData` arrays of shape `(num_envs, …)`.
2. **Scene Layer** — `Scene` owns named `Entity` objects (robot, ball,
   obstacles), each with a precompiled `MjSpec`. `SceneEntityCfg` resolves
   name patterns (`body_names=".*_knee"`) into global MuJoCo indices once at
   startup so term functions can index tensors directly at runtime.
3. **Task Layer** — `ManagerBasedRlEnv` orchestrates managers that own
   declarative *terms*. Mjlab's standard managers are:
   - `ActionManager` (joint position / IK actions)
   - `ObservationManager` (named obs groups, e.g. `actor`, `critic`)
   - `RewardManager` (weighted reward terms)
   - `TerminationManager` (per-env done flags)
   - `EventManager` (reset / startup / interval hooks: domain randomization)
   - `CommandManager` (sampled commands like target velocity, gait phase)
   - `CurriculumManager` (step-keyed schedule changes)

The whole declarative-term pattern is **inherited** by Colosseum: every
Colosseum-specific manager follows the same `*TermCfg → *Term → *Manager`
shape so it composes seamlessly with the mjlab pieces.

---

## 1.3 Colosseum-specific environment: `ColosseumEnv`

File: `src/colosseum/envs/colosseum_env.py`

Single class used by every modern Colosseum task. Configuration is
`ColosseumEnvCfg`, which extends `ManagerBasedRlEnvCfg` with three optional
dicts:

```python
@dataclass(kw_only=True)
class ColosseumEnvCfg(ManagerBasedRlEnvCfg):
    encoders:     dict[str, RmaTermCfg]         = field(default_factory=dict)
    abstractions: dict[str, AbstractionTermCfg] = field(default_factory=dict)
    constraints:  dict[str, ConstraintTermCfg]  = field(default_factory=dict)
    use_depth_camera: bool = False
    viz_callbacks: list[tuple[str, Callable]] = field(default_factory=list)
```

`ColosseumEnv.load_managers()` instantiates **only** the extra managers whose
config dict is non-empty:

| Field | Manager class | Purpose |
|---|---|---|
| `encoders` | `RmaManager` | Two-phase RMA encoders (privileged + adaptation). |
| `abstractions` | `AbstractionManager` | Cached planner-derived signals (e.g. grid-direction maps). |
| `constraints` | `ConstraintManager` | "Constraints as Terminations" soft penalties. |

The legacy classes (`AbstractionBasedEnv`, `ConstraintBasedEnv`,
`ConstraintRmaEnv`, `RmaBasedEnv`, `ViewerCompatibleEnv`) still live in
`src/colosseum/envs/` and are re-exported from `colosseum.envs.__init__` for
backward compatibility, but new tasks always use `ColosseumEnv`/`ColosseumEnvCfg`.

---

## 1.4 The three Colosseum managers

All three follow mjlab's `ManagerBase` / `ManagerTermBase` pattern: a config
dict `name → *TermCfg` is turned into runtime *terms* by `_prepare_terms()`,
and the manager exposes a stable `compute()`/`reset()`/`update()` lifecycle.

### 1.4.1 `RmaManager` — `src/colosseum/managers/rma_manager.py`

Runs Rapid Motor Adaptation (Phase 1 = privileged encoder; Phase 2 =
sensor-conditioned adaptation encoder). Key types:

- `RmaTermCfg(ABC)` — fields: `privileged_obs_group: str`,
  `adaptation_obs_group: str | None`, `latent_dim`, `latent_noise_std`.
- `RmaTerm(ManagerTermBase)` — owns both encoders, exposes
  `encode_privileged`, `encode_adaptation`, optional `update`/`reset`/
  `compute_loss`/`get_adaptation_mask`/`get_current_adaptation_obs`.
- `RmaManager(ManagerBase)` — concatenates per-term latents in declaration
  order to give the actor a stable input layout.

The dribbling term (`src/colosseum/research/dribbling/rma_terms.py`,
`DribblingRmaTerm`) is the canonical implementation and uses
`PrivilegedEncoder` (`src/colosseum/algorithm/networks/privileged_encoder.py`)
for Phase 1 and `DepthEncoder + BallHead + ObstacleHead`
(`src/colosseum/research/dribbling/encoders.py`) for Phase 2.

### 1.4.2 `AbstractionManager` — `src/colosseum/managers/abstraction_manager.py`

Provides cached planner-derived guidance signals (cost maps, harmonic
potential fields, Sokoban-style sub-goals). Key types:

- `AbstractionSettings` — immutable dataclass identifying a "planning
  problem" (e.g. `(goal_cell, map_id)`); change-detection drives rebuild.
- `AbstractionTerm(ABC)` — `_update_settings`, `_maybe_rebuild`,
  `_update_signals`. The term *does not* expose a fixed signal interface;
  consumers (commands, rewards, observations) call its task-specific methods
  directly via `manager.get_term(name)`.
- Concrete terms live under `src/colosseum/mdp/abstraction/maze/`:
  - `grid_abstraction.py`: harmonic / Dijkstra direction field on a maze.
  - `sokoban_grid_abstraction.py`: Sokoban-style sub-goal planner using
    `sokoban_solver.py` (and the `up-symk` PDDL planner from `pixi.toml`).
- `NullAbstractionManager` is a no-op stub for envs that disable abstractions.

### 1.4.3 `ConstraintManager` — `src/colosseum/managers/constraint_manager.py`

Implements **Constraints-as-Terminations** ([CaT, arXiv:2403.18765](https://arxiv.org/abs/2403.18765)).
Each term returns a per-env (optionally per-dim) violation tensor. The
manager:

1. Polyak-averages a running max of each violation,
2. converts the normalized violation into a per-env termination probability
   `p ∈ [0, 1]` with a per-term cap `max_p`,
3. returns the per-env max probability over all terms.

In `ColosseumEnv.step()`, the returned probability is used to (a) scale
rewards by `(1 - p)` and (b) softly terminate (the env returns the
probability as a float `terminated`, then the bootstrap logic in PPO handles
it consistently).

Constraint functions live in `src/colosseum/mdp/constraints.py` (joint
torque/velocity/range/limits, base orientation, sensor-based contacts, ball
proximity, …). The CaT helper class itself is `class CaT` in the same file
as the manager.

---

## 1.5 MDP helpers (`src/colosseum/mdp/`)

These are **pure functions** that any task can plug into mjlab managers:

- `mdp/observations.py` — robot-agnostic obs (`compute_projected_gravity`,
  `agent_pos`, `goal_pos`, `agent_to_goal_vector`, …).
- `mdp/rewards.py` — generic rewards (`flat_orientation`, `pose_deviation`,
  `feet_distance_penalty`, `swing_phase_schedule`, `stance_phase_schedule`).
- `mdp/ball_rewards.py` — ball-aware rewards reused by dribbling/soccer-maze.
- `mdp/constraints.py` — CaT constraint functions.
- `mdp/abstraction/maze/*` — planners and abstraction terms.

Each task can also have its own task-local `mdp/` package (e.g.
`tasks/dribbling/mdp/`) for task-specific terms it does not want to expose
globally.

---

## 1.6 Tasks (`src/colosseum/tasks/<task>/`)

Each task is a Python sub-package with a fixed shape:

```
src/colosseum/tasks/<task>/
├── __init__.py                ← imports its config submodules to fire
│                                @register_task decorators on import
├── config/
│   ├── __init__.py            ← imports per-robot config sub-package(s)
│   └── t1_23dof/              ← per-robot variant
│       ├── __init__.py        ← imports the *_cfg.py top-level config
│       ├── t1_<task>_cfg.py   ← scene_cfg, env_cfg, viewer, sim, register_task
│       ├── algo_cfg.py        ← per-task PpoConfig / RmaPPOConfig / DaggerPpoConfig
│       ├── observation_cfg.py ← actor_terms / critic_terms / privileged_*
│       ├── reward_cfg.py      ← rewards: dict[str, RewardTermCfg]
│       ├── event_cfg.py       ← events: dict[str, EventTermCfg]
│       ├── cact_cfg.py        ← Commands + Actions + Curriculum + Terminations
│       └── constraint_cfg.py  ← (only if the task uses CaT)
└── mdp/                       ← task-local term functions, optional
    ├── observations.py / rewards.py / terminations.py / curriculums.py
    ├── events.py
    └── *_command.py            ← custom CommandTerm subclasses
```

`src/colosseum/tasks/__init__.py` walks the package with `pkgutil.iter_modules`
so every task self-registers when `colosseum.tasks` is imported anywhere
(`scripts/train.py`, `scripts/play.py` and `config/values/task.py` all do
this).

The four tasks shipped today are:

| `task_id` | Folder | Robot | Highlights |
|---|---|---|---|
| `t1-velocity` | `tasks/velocity/` | T1 23 DOF | Plain velocity tracking (UniformVelocityCommand), no extras. |
| `t1-dribbling` | `tasks/dribbling/` | T1 23 DOF | Ball + obstacles + RMA encoder + CaT constraint + DAgger variant. |
| `t1-maze` | `tasks/maze/` | T1 23 DOF | Maze navigation + grid abstraction (harmonic field). |
| `t1-soccer-maze` | `tasks/soccer_maze/` | T1 23 DOF | Dribble through maze: ball + Sokoban abstraction + CaT. |

---

## 1.7 Algorithms (`src/colosseum/algorithm/`)

All algorithms inherit from `BaseAlgorithm` (`base_algorithm.py`), an
`ABC` that:

- Owns `self.env`, `self.actor`, `self.actor_obs_normalizer`.
- Exposes a uniform checkpoint interface (`save`, `load`, `_load_checkpoint`,
  `_save_checkpoint`, `attach_metadata`, `configure_checkpointing`).
- Standardizes logging: `_log_training_metrics`, episode-success tracking,
  Rich console output, distributed (torchrun) primitives.
- Defines an `export_onnx(path)` hook used by `scripts/export_onnx.py`.

Concrete algorithms:

| Algorithm | File | Inherits | What's added |
|---|---|---|---|
| `PPO` | `ppo.py` | `BaseAlgorithm` | RSL-RL-style on-policy PPO with adaptive-KL LR, single optimizer, GAE, optional clipped value loss. |
| `RmaPPO` | `rma_ppo.py` | `PPO` | Wires `env.unwrapped.rma_manager` into PPO: widens actor input, optimizes encoder params jointly in Phase 1, runs Phase 2 adaptation regression (`_train_phase2`). |
| `DaggerPPO` | `dagger_ppo.py` | `PPO` | Adds an annealed imitation-MSE loss against a frozen `TeacherPolicy` whose obs is a contiguous suffix of the student's obs. |
| `DaggerRmaPPO` | `dagger_rma_ppo.py` | `RmaPPO` | Same DAgger trick but compatible with the RMA actor input layout. |

All four register themselves via the `@register_algorithm("name", config_class=…)`
decorator from `colosseum.config.types.algorithm`. The decorator stores
`(impl_class, config_class)` keyed by lowercase name in
`_ALGORITHM_REGISTRY`. Resolution at runtime is done in `scripts/train.py` and
`scripts/play.py` via the `target` field of the algo config (`module:class`),
not via the registry — so registering is not strictly required, but the
registry is what `play.py` and `export_onnx.py` rely on for reverse lookups.

### Networks (`algorithm/networks/`)

- `ppo_networks.py` — `PpoActor` (Gaussian policy, state-independent learnable
  std, no tanh squash, RSL-RL init), `PpoValueNet`, generic `Network` MLP
  backbone, `_resolve_activation` helper.
- `privileged_encoder.py` — fixed-architecture MLP for Phase 1 RMA: `Linear → 32 → ELU → Linear → latent → LayerNorm`.
- `teacher_policy.py` — wraps a frozen pretrained checkpoint for DAgger: it
  slices the student obs (`obs[:, obs_start_idx:]`), applies its own
  observation normalizer, and returns deterministic teacher actions.

### Utilities (`algorithm/utils/`)

- `rollout_buffer.py` — pre-allocated `[T, N, dim]` storage with optional
  `extras` and `privileged_obs_dims` slots (used by DAgger and RMA
  respectively); GAE in-place, mini-batch generator with shuffling.
- `normalization.py` — `EmpiricalNormalization` (RSL-RL-style running
  mean/var) and a no-op `IdentityNormalizer`.

---

## 1.8 Configs (`src/colosseum/config/`)

```
config/
├── types/
│   ├── algorithm.py    ← AlgorithmConfig, PpoConfig, RmaPPOConfig,
│   │                     DaggerPpoConfig + register_algorithm decorator
│   ├── task.py         ← TaskConfig + @register_task decorator + _TASK_REGISTRY
│   ├── networks.py     ← NetworkConfig, PpoActorConfig, PpoCriticConfig
│   ├── logger.py       ← LoggerConfig
│   └── experiment.py   ← BaseExperimentConfig, TrainConfig (tyro entry)
└── values/
    ├── algorithm.py    ← PPO_DEFAULT (a global PpoConfig)
    ├── logger.py       ← WANDB / DISABLED defaults
    └── task.py         ← DEFAULTS = _TASK_REGISTRY (live view of @register_task)
```

`TrainConfig` is the top-level dataclass parsed by `tyro` in
`scripts/train.py`. It pulls a task via `tyro.extras.subcommand_type_from_defaults`
keyed on `_TASK_REGISTRY`, so `pixi run train task:t1-velocity` directly
selects a registered task subclass.

Algo configs use `target: str` of the form `"module:class"`. The training
script imports that module and instantiates the class — this is how a task
chooses its algorithm without touching the script.

---

## 1.9 Scripts (`src/colosseum/scripts/`)

Mapped to `pixi` task names in `pixi.toml`:

| `pixi run …` | Script | Purpose |
|---|---|---|
| `train` | `train.py` | Custom PPO training loop (the default). Handles tyro parsing, distributed (`torchrun`) bootstrap, W&B and Loguru, checkpoint resume / warm start, signal handling (Ctrl-C → save). |
| `train-rsl-rl` | `train_rsl_rl.py` | Alternative trainer using mjlab's built-in RSL-RL `MjlabOnPolicyRunner`. |
| `train-mjlab` | `train_mjlab.py` | Calls `mjlab.scripts.train` for tasks defined in mjlab itself. |
| `train-phase2` | `research/dribbling/scripts/train_phase2.py` | RMA Phase 2: freeze actor + privileged encoder, train adaptation encoder. |
| `pipeline-dribbling` | `research/dribbling/scripts/pipeline_dribbling.py` | Multi-stage curriculum pipeline (Phase 1 + Phase 2 per stage). |
| `eval-dribbling` | `research/dribbling/scripts/evaluate_dribbling.py` | Evaluation protocol used in the dribbling paper. |
| `obstacle-ablation` | `research/dribbling/scripts/obstacle_ablation.py` | Ablation study for obstacle presence. |
| `play` | `play.py` | Runs a checkpoint (or `zero`/`random`/`onnx` agent) in a viewer. Auto-discovers latest run. |
| `export-onnx` | `export_onnx.py` | Export an actor checkpoint to ONNX and register it in `models/registry.yaml`. |
| `model-registry` | `model_registry_cli.py` | List / set-default / remove ONNX entries in the registry. |
| `cleanup-run` | `cleanup_run.py` | Remove a run from disk + W&B. |
| `docs` / `build-docs` | (mkdocs) | Build / serve the documentation site. |

---

## 1.10 Robots and assets

### Robot: Booster T1 (`src/colosseum/robots/t1_23dof/`)

Files:

- `xmls/T1_23dof.xml` (and any auxiliary mesh assets) — the MJCF model.
  No `<actuator>` block is used at XML level — the actuator config is added
  in Python so it can be tuned and reused across tasks.
- `constants.py` — joint names, home keyframe (`HOME_QPOS`), action scale,
  foot geom names, base body name, head-camera intrinsics, helper functions
  `get_spec()` / `get_spec_with_head_camera()`, the canonical
  `EntityArticulationInfoCfg ARTICULATION` and the entry point
  `get_robot_cfg(foot_self_collision, with_head_camera)` that returns a ready
  `EntityCfg`.
- `actuators.py` — per-joint `ActuatorCfg` objects (`T1_ACTUATOR_HIP_PITCH`,
  `T1_ACTUATOR_KNEE`, `T1_ACTUATOR_ANKLE_PITCH`, …) and the
  `compute_pd_gains()` helper (Unitree-G1 method: natural frequency + damping
  ratio).
- `collisions.py` — `FEET_ONLY_COLLISION`, `FEET_SELF_COLLISION`, etc.
- `sensors.py` — pre-built sensor configs (`FEET_GROUND_CONTACT_SENSOR`,
  `FOOT_HEIGHT_SCAN`, `FOOT_BALL_CONTACT_SENSOR`, `WALL_COLLISION_SENSOR`,
  `HEAD_DEPTH_SENSOR_TRAIN`, `HEAD_RGBD_SENSOR`, `SELF_COLLISION_SENSOR`, …).

Note: the legacy `booster_t1` package referenced in CLAUDE.md was renamed
to `t1_23dof`. There is also `robots/ant/` for a smaller-scale playground.

### Assets: Soccer ball (`src/colosseum/assets/ball/`)

- `ball_spec.py` — `get_ball_cfg()` returning an `EntityCfg` for a free
  floating ball with manufacturer-realistic mass and friction; constants
  `BALL_MASS` and `BALL_FRICTION` are exported for use as observation params.

---

## 1.11 Utilities (`src/colosseum/utils/`)

- `checkpoint.py` — `resolve_checkpoint(checkpoint, run_name, log_dir)` used
  by every script that needs to pick a `.pt` file (priority: explicit path →
  `<log_dir>/<run_name>/checkpoints/` → most recently modified run).
- `model_registry.py` — read/write `models/registry.yaml`, sync the
  `models/<task>/default.onnx` symlink.
- `export.py` — `export_policy_to_onnx(config, ckpt, out)`, glue between an
  experiment config and `BaseAlgorithm.export_onnx`.
- `logger.py` — Loguru/W&B setup, run-name generation, episode-metric
  extraction, Rich training panels.
- `torch.py` — `set_seed`, `get_device`, `get_obs_dims` helpers.
- `grid_frame.py` — `GridFrame` for grid↔world coordinate conversion (used
  by abstractions and the maze terrain importer).
- `isaaclab/{array, dict, math, string}.py` — small utilities ported from
  Isaac Lab (e.g. `quat_apply`, `quat_apply_inverse`).
- `deploy/{metrics, synced_array}.py` — helpers for hardware deployment
  measurement / synchronization.

---

## 1.12 Research projects (`src/colosseum/research/`)

These are paper-specific extensions that re-use the generic infrastructure.

- `research/dribbling/` — RMA term (`rma_terms.py`), depth encoder
  (`encoders.py`), and the four scripts wired into pixi (Phase 2 training,
  curriculum pipeline, evaluation protocol, obstacle ablation).
- `research/soccer_maze/` — benchmark configuration, training-benchmark
  driver, evaluation, plotting.

If you want to read the published context, see `docs/research/dribbling.md`
and `docs/research/soccer-maze.md`.

---

## 1.13 Documentation (`docs/`)

The repo ships its own MkDocs Material site (`mkdocs.yml`):

- `docs/colosseum/getting_started.md`, `setup.md`, `cartpole.md`,
  `dribbling_tutorial.md`, `internals.md`, `training.md`, `deployment.md`.
- `docs/mjlab/{overview, scene_simulation, task_layer}.md` — mjlab background.
- `docs/research/{dribbling, soccer-maze}.md` — paper write-ups.
- `docs/architecture/` — **this report**.

`pixi run docs` serves the site on `http://localhost:8000`; `pixi run
build-docs` writes to `site/`.

---

## 1.14 Model registry (`models/`)

- `models/registry.yaml` — top-level YAML with `<task>: {default, models: {…}}`
  entries. Each model entry has `file`, `run`, `step`, `created`.
- `models/<task>/default.onnx` — symlink to the current default model.
- `models/<task>/<name>/<task>_<algo>_<name>.onnx` — actual ONNX files,
  one folder per registered model.

The CLI (`pixi run model-registry list|set-default|remove`) and the play
script (`pixi run play … --onnx default|v1|<path>`) both go through
`utils/model_registry.py`.

---

## 1.15 Glance at one fully wired example: `t1-velocity`

This is the smallest task that exercises the whole pipeline end-to-end.
Its files are:

- `tasks/velocity/__init__.py` — imports the per-robot config to register.
- `tasks/velocity/config/__init__.py` — imports `t1_23dof`.
- `tasks/velocity/config/t1_23dof/__init__.py` — imports `t1_velocity_cfg.py`.
- `tasks/velocity/config/t1_23dof/t1_velocity_cfg.py` — defines `scene_cfg`,
  `viewer_cfg`, `sim_cfg`, `booster_t1_velocity_env_cfg`, registers
  `T1VelocityTask` with `@register_task("t1-velocity")`.
- `tasks/velocity/config/t1_23dof/cact_cfg.py` — actions (`JointPositionActionCfg`),
  commands (`UniformVelocityCommandCfg`), curriculum (`commands_vel`),
  terminations (`time_out`, `bad_orientation`).
- `tasks/velocity/config/t1_23dof/observation_cfg.py` — `actor_terms`
  (proprio + command), `critic_terms` (= actor + base_lin_vel + foot
  metrics), grouped into `actor`/`critic` `ObservationGroupCfg`.
- `tasks/velocity/config/t1_23dof/reward_cfg.py` — track_linear/angular
  velocity, `flat_orientation`, `variable_posture`, foot air-time/clearance/
  swing-height/slip/soft-landing, `self_collision_cost`, action_rate, joint
  limits.
- `tasks/velocity/config/t1_23dof/event_cfg.py` — `reset_root_state_uniform`,
  `reset_joints_by_offset`, `push_robot` (interval), `geom_friction`,
  `encoder_bias`, `body_com_offset` (startup domain randomization).
- `tasks/velocity/config/t1_23dof/algo_cfg.py` — `booster_t1_ppo_cfg()`
  returning a `PpoConfig` plus an alternative `booster_t1_rsl_rl_runner_cfg()`
  for the RSL-RL trainer.

There are **no encoders, no abstractions, no constraints**, so `ColosseumEnv`
constructs only the standard mjlab managers — but the same task could opt
into any of the three by adding the corresponding non-empty dict to
`ColosseumEnvCfg`. That is exactly what `t1-dribbling`, `t1-maze`, and
`t1-soccer-maze` do.

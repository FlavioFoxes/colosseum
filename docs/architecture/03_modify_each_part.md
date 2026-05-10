# 3. Modification recipes

Each section below is a self-contained "what files do I touch?" recipe for a
single kind of change. Worked examples live in `04_examples.md`; here we
focus on the **mechanical** steps.

If a recipe ends with a "test it" step, the smallest sanity-check is always:

```
pixi run train task:<task-id> --task.algo-cfg.learning-steps 24000 --logger disabled
```

(That fits in well under a minute and exercises the full pipeline once.)

---

## 3.1 Add a new task

Goal: register a new `task_id` discoverable by `pixi run train task:<task_id>`.

1. **Pick a folder name** under `src/colosseum/tasks/<task>/` (use snake_case).
2. **Mirror the velocity layout**:

   ```
   src/colosseum/tasks/<task>/
     __init__.py
     config/__init__.py
     config/<robot>/__init__.py
     config/<robot>/t1_<task>_cfg.py
     config/<robot>/observation_cfg.py
     config/<robot>/reward_cfg.py
     config/<robot>/event_cfg.py
     config/<robot>/cact_cfg.py
     config/<robot>/algo_cfg.py
     mdp/                            (only if you need task-local term funcs)
   ```

3. **Wire auto-import**:
   - `tasks/<task>/__init__.py` → `import colosseum.tasks.<task>.config.<robot>.t1_<task>_cfg`
     (wrap in a `try / except ImportError: pass` so optional deps don't break
     other tasks).
   - `tasks/<task>/config/__init__.py` → `from .<robot> import *  # noqa: F401`
     or simply nothing if the previous import path is enough.
   - `tasks/<task>/config/<robot>/__init__.py` → `from .t1_<task>_cfg import *  # noqa: F401`.
4. **Define the env factory** in `t1_<task>_cfg.py`:
   - `scene_cfg(play=False) -> SceneCfg` (entities + sensors + terrain).
   - `viewer_cfg() -> ViewerConfig`.
   - `sim_cfg() -> SimulationCfg`.
   - `make_env_cfg(play=False, …) -> ColosseumEnvCfg` collecting all the
     dicts: `observations`, `actions`, `commands`, `events`, `rewards`,
     `terminations`, `curriculum`, optional `encoders` / `abstractions` /
     `constraints`, plus `decimation`, `episode_length_s`, `metrics={}`.
5. **Register** at module level:

   ```python
   from colosseum.config.types.task import TaskConfig, register_task

   @register_task("my-task")
   @dataclass(frozen=True)
   class MyTask(TaskConfig):
     name: str = "my-task"
     env: ColosseumEnvCfg = field(default_factory=make_env_cfg)

     @property
     def train_env_cfg(self): return self.env
     @property
     def play_env_cfg(self):  return make_env_cfg(play=True)
     @property
     def algo_cfg(self):      return my_task_ppo_cfg()
   ```

6. **Verify** the task is registered by running
   `pixi run python -c "import colosseum.tasks; from colosseum.config.types.task import list_tasks; print(list_tasks())"`.
7. **Smoke-train**:
   ```
   pixi run train task:my-task --task.env.scene.num-envs 64 --logger disabled \
       --task.algo-cfg.learning-steps 24000
   ```

---

## 3.2 Add a new algorithm

Goal: write a new training loop and make it selectable via the algo config's
`target` field.

1. **Create the impl** in `src/colosseum/algorithm/<name>.py`:
   - Subclass `BaseAlgorithm` directly (custom loop) or one of the existing
     classes (`PPO`, `RmaPPO`) to inherit collection/learning infrastructure.
   - Implement / override the methods you need: `train`, `_collect_rollout`,
     `_learning_step`, `save`, `load`, `_build_networks`, `_build_optimizers`,
     `_build_rollout_buffer`, `_compose_actor_input`, `_eval_get_action`,
     `export_onnx`.
2. **Define the config dataclass** in `src/colosseum/config/types/algorithm.py`
   (or a new module imported there). It should:
   - Subclass `AlgorithmConfig` (or `PpoConfig` if you reuse PPO knobs).
   - Set `name`, `target` (`"colosseum.algorithm.<name>:<Class>"`).
   - Add hyper-parameter fields with sensible defaults.
3. **Register** the algorithm:

   ```python
   from colosseum.config.types.algorithm import register_algorithm

   @register_algorithm("my_algo", config_class=MyAlgoConfig)
   class MyAlgo(BaseAlgorithm):
       ...
   ```

4. **Expose it from a task**: add a builder function in the relevant
   `tasks/<task>/config/<robot>/algo_cfg.py` that returns a `MyAlgoConfig`
   instance, and have `<TaskConfig>.algo_cfg` return it.
5. **(Optional)** add a default to `src/colosseum/config/values/algorithm.py`
   so `pixi run train` without an explicit task still works.
6. **Smoke-train** as in 3.1; add `--task.algo-cfg.<field>` to override.

---

## 3.3 Add a reward term

Goal: add a new entry to `rewards: dict[str, RewardTermCfg]`.

1. **Decide where to put the function**:
   - **Generic / robot-agnostic** → `src/colosseum/mdp/rewards.py`
     (or `mdp/ball_rewards.py` for ball-related).
   - **Task-local** → `src/colosseum/tasks/<task>/mdp/rewards.py`.
2. **Write the function** with the canonical signature:

   ```python
   def my_reward(
       env: ManagerBasedRlEnv,
       asset_cfg: SceneEntityCfg,
       std: float,
       command_name: str,
       ...
   ) -> torch.Tensor:
       """Returns a (num_envs,) reward tensor."""
       asset = env.scene[asset_cfg.name]
       ...
       return reward
   ```

   Use `asset_cfg.body_ids` / `joint_ids` / `site_ids` (already resolved at
   init time) to index into `asset.data.*`.
3. **Register it** in `tasks/<task>/config/<robot>/reward_cfg.py`:

   ```python
   rewards["my_reward"] = RewardTermCfg(
       func=my_reward, weight=0.5,
       params={
         "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
         "std": 0.5,
         "command_name": "twist",
       },
   )
   ```

4. Run a short job to check the new key appears in the W&B episode-reward
   panel: `Episode_Reward/my_reward`.

---

## 3.4 Add an observation term / observation group

Goal: extend `observations` with a new term, and (optionally) introduce a new
named group consumable by RMA encoders or critics.

1. **Implement the term function** in `mdp/observations.py` (generic) or
   the task's `mdp/observations.py`. Signature:

   ```python
   def my_obs(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
       return ...   # shape (num_envs, dim)
   ```

2. **Add it** to `actor_terms` and/or `critic_terms` in
   `tasks/<task>/config/<robot>/observation_cfg.py`:

   ```python
   actor_terms["my_obs"] = ObservationTermCfg(
       func=my_obs, params={...}, noise=Unoise(n_min=-0.01, n_max=0.01))
   ```

3. **Add a new group** if you need privileged observations consumed only by
   the critic, an RMA encoder, or a custom callback:

   ```python
   privileged_my_terms = {"my_priv": ObservationTermCfg(func=my_priv)}
   observations["privileged_my"] = ObservationGroupCfg(
       terms=privileged_my_terms,
       concatenate_terms=True,
       enable_corruption=False,
   )
   ```

4. **Make sure the group is referenced** by whatever consumes it (RMA term's
   `privileged_obs_group`, critic with `**privileged_my_terms`, etc.).

---

## 3.5 Add an event term (domain randomization / reset hook)

Goal: extend `events: dict[str, EventTermCfg]`.

1. **Pick a function**:
   - mjlab built-ins: `mjlab.envs.mdp.events`, `mjlab.envs.mdp.dr` (e.g.
     `reset_root_state_uniform`, `push_by_setting_velocity`,
     `geom_friction`, `body_com_offset`, `encoder_bias`).
   - Your own: write it in `tasks/<task>/mdp/events.py` (signature:
     `def f(env, env_ids, **params)`).
2. **Register** in `tasks/<task>/config/<robot>/event_cfg.py`:

   ```python
   events["my_event"] = EventTermCfg(
       func=my_event, mode="reset",     # or "startup" / "interval"
       interval_range_s=(2.0, 5.0),     # only for "interval"
       params={"asset_cfg": SceneEntityCfg("robot"), ...},
   )
   ```

3. The mode controls when it fires:
   - `startup` — once at env construction (use for permanent randomization).
   - `reset` — every time the env (or selected envs) reset.
   - `interval` — at random intervals (also requires `interval_range_s`).

---

## 3.6 Add a termination term

Goal: extend `terminations: dict[str, TerminationTermCfg]`.

1. **Implement** a function returning a `(num_envs,) bool` tensor.
   - Mjlab built-ins: `time_out`, `bad_orientation` (from
     `mjlab.envs.mdp.terminations`).
   - Task-local: e.g. `tasks/maze/mdp/terminations.py::arrived_at_goal`.
2. **Register** in `cact_cfg.py`:

   ```python
   terminations["my_term"] = TerminationTermCfg(
       func=my_term,
       params={...},
       time_out=False,    # True only for time-out (truncation), not failure
   )
   ```

3. The first termination term named `arrived_at_goal` is recognised by
   `BaseAlgorithm.update_episode_counts` for rolling success-rate tracking.
   Use that exact name if your task has a clear "success" condition.

---

## 3.7 Add a curriculum term

Goal: schedule a hyperparameter / config change at fixed env-step thresholds.

1. **Pick or write a curriculum function** that takes
   `(env, env_ids, ...)` and mutates the relevant config slot at the right
   moment. Examples in the repo:
   - `mjlab.tasks.velocity.mdp.curriculums.commands_vel`: widens
     `command_manager.get_term(name).cfg.ranges`.
   - `tasks/maze/mdp/curriculums.py::base_velocity_curriculum`,
     `wall_collision_termination_curriculum`.
2. **Register** in `cact_cfg.py`:

   ```python
   curriculum["my_stage"] = CurriculumTermCfg(
       func=my_curriculum,
       params={"velocity_stages": [{"step": 0, "value": 0.1}, ...]},
   )
   ```

The `CurriculumManager` calls each term every step; the term itself decides
when to act based on `env.common_step_counter`.

---

## 3.8 Add a command term

Goal: introduce a new sampled command (e.g. a target position, a gait phase).

1. **Subclass** `mjlab.managers.CommandTerm` and `CommandTermCfg` if you
   need a fully custom term. Examples:
   - `tasks/dribbling/mdp/ball_velocity_command.py::BallVelocityCommand`
   - `tasks/dribbling/mdp/gait_phase_command.py::GaitPhaseCommand`
   - `tasks/dribbling/mdp/obstacle_commands.py::ObstacleCommand`
   - `mdp/abstraction/maze/abstraction_velocity_command.py::AbstractionVelocityCommand`
2. **Implement** at minimum:
   - `__init__(self, cfg, env)` allocates internal tensors `(num_envs, …)`.
   - `_resample_command(env_ids)` samples new targets for the given ids.
   - `_update_command()` recomputes the world-frame command per step.
   - `command` property returns the current per-env tensor.
3. **Register** in `cact_cfg.commands["my_cmd"] = MyCmdCfg(…)`.
4. **Consume** in observations (`generated_commands` with `command_name`),
   rewards (e.g. `track_linear_velocity` taking `command_name="my_cmd"`),
   and (optionally) curricula.

---

## 3.9 Add an RMA encoder term

Goal: put new privileged information through a learned latent for the actor.

1. **Pick the privileged obs group** the encoder will read. If the group
   does not yet exist, add one in `observation_cfg.py` (see 3.4) with
   `enable_corruption=False`.
2. **Pick the adaptation obs group** (or `None` if only Phase 1). For depth
   you can reuse the dribbling `"depth_frames"` group; otherwise add it.
3. **Subclass `RmaTermCfg` and `RmaTerm`** in
   `src/colosseum/research/<paper>/rma_terms.py` (or anywhere; place near
   the consumer). Implement at minimum:

   ```python
   @dataclass(kw_only=True)
   class MyRmaTermCfg(RmaTermCfg):
       privileged_obs_group: str = "privileged_my"
       adaptation_obs_group: str | None = None
       latent_dim: int = 32
       latent_noise_std: float = 0.0
       def build(self, env): return MyRmaTerm(self, env)

   class MyRmaTerm(RmaTerm):
       def __init__(self, cfg, env):
           super().__init__(cfg, env)
           # build self._priv_encoder (input_dim from
           # env.observation_manager.group_obs_dim[cfg.privileged_obs_group][0])
           # build self._adapt_encoder if adaptation_obs_group is set

       @property
       def privileged_encoder(self): return self._priv_encoder
       @property
       def adaptation_encoder(self): return getattr(self, "_adapt_encoder", None)

       def encode_privileged(self, obs_dict):
           x = obs_dict[self.cfg.privileged_obs_group]
           return self._priv_encoder(x)

       def encode_adaptation(self, obs_dict):
           x = obs_dict[self.cfg.adaptation_obs_group]
           return self._adapt_encoder(x)

       # Optional: update / reset / get_adaptation_mask /
       # get_current_adaptation_obs / compute_loss / extra_state_dict
   ```

4. **Register** in your task's `make_env_cfg`:

   ```python
   cfg.encoders = {"my_enc": MyRmaTermCfg(latent_dim=16)}
   ```

5. **Switch the algorithm** to `RmaPPO` (or `DaggerRmaPPO`):
   - Update `algo_cfg` to return `RmaPPOConfig(..., target="colosseum.algorithm.rma_ppo:RmaPPO")`.
6. **Phase 2 training** is then triggered by the `train-phase2` script.

---

## 3.10 Add an abstraction term

Goal: introduce a new planner-derived signal cached at the env level.

1. **Subclass** `AbstractionTermCfg` and `AbstractionTerm` in
   `src/colosseum/mdp/abstraction/<topic>/<term>.py` (e.g. mirror
   `grid_abstraction.py`). Implement:
   - A frozen `MyAbstractionSettings(AbstractionSettings)` dataclass with
     just the fields whose change should trigger a rebuild.
   - `_update_settings(env_ids)` reads current env state into a per-env
     settings registry.
   - `_maybe_rebuild(env_ids)` runs the planner for any new settings and
     stores the cached cost/direction/sub-goal data.
   - `_update_signals(env_ids)` reads current robot/ball pos and looks up
     the cached signal — fast inner-loop work.
   - Public methods that consumers can call (e.g. `get_direction(env_ids)`,
     `get_subgoal(env_ids)`).
2. **Register** in `make_env_cfg`:
   ```python
   cfg.abstractions = {"my_abs": MyAbstractionTermCfg(...)}
   ```
3. **Consume** from a command/reward/observation function via
   `env.abstraction_manager.get_term("my_abs").get_direction(...)`.

---

## 3.11 Add a constraint term (CaT)

Goal: add a soft termination probability for some violation metric.

1. **Implement** `def my_constraint(env, **params) -> torch.Tensor` in
   `src/colosseum/mdp/constraints.py` (or your task's constraint module).
   It must return `(num_envs,)` or `(num_envs, num_dims)` where positive
   values mean violation magnitude.
2. **Register** in `tasks/<task>/config/<robot>/constraint_cfg.py`:

   ```python
   constraints["my_cstr"] = ConstraintTermCfg(
       func=my_constraint,
       max_p=0.5,
       params={"limit": 1.0, "asset_cfg": SceneEntityCfg("robot")},
   )
   ```

3. **Wire** in `make_env_cfg`: `cfg.constraints = constraints`.
4. The manager handles Polyak max + reward scaling + soft termination
   automatically; PPO does not need any change.

---

## 3.12 Add or modify a robot

Goal: support a new humanoid (or extend T1).

1. **Drop the MJCF** under `src/colosseum/robots/<robot>/xmls/<robot>.xml`
   along with any meshes.
2. **Author the per-joint actuator configs** in
   `src/colosseum/robots/<robot>/actuators.py` (use `compute_pd_gains`
   helper if you have natural-frequency / damping-ratio targets).
3. **Author collisions** in `collisions.py` (`FEET_ONLY_COLLISION`,
   self-collision rules, etc.).
4. **Author sensors** in `sensors.py` (contact sensors, height scans, depth
   cameras …).
5. **Compose** in `constants.py`:
   - `JOINT_NAMES`, `HOME_QPOS`, `ACTION_SCALE`, `FOOT_GEOM_NAMES`,
     `BASE_BODY_NAME`, `XML` path.
   - `ARTICULATION = EntityArticulationInfoCfg(actuators=(...))`.
   - `def get_spec()`/`get_spec_with_head_camera()` returning `MjSpec`.
   - `def get_robot_cfg(...)` returning a fully populated `EntityCfg`.
6. **Use it** from any task by replacing the import:

   ```python
   from colosseum.robots.<robot>.constants import get_robot_cfg, ACTION_SCALE
   ```

7. **Add a per-robot variant** to existing tasks if desired:
   `tasks/<task>/config/<robot>/...` with all five `*_cfg.py` files (PD
   gains, limits, and reward stds usually need re-tuning per robot).

---

## 3.13 Add a scene asset (e.g. a new ball, cone, goal)

Goal: spawn a free / static body alongside the robot.

1. **Create the spec module** under `src/colosseum/assets/<asset>/<asset>_spec.py`:

   ```python
   def get_<asset>_cfg() -> EntityCfg:
       return EntityCfg(
           init_state=EntityCfg.InitialStateCfg(pos=(0,0,0.1)),
           spec_fn=load_<asset>_spec,
           articulation=...
       )
   ```

2. **Hook into a task's `scene_cfg`**:

   ```python
   entities = {"robot": get_robot_cfg(), "<asset>": get_<asset>_cfg()}
   ```

3. **Add reset / push events** if you want the asset randomized at episode
   start (`tasks/<task>/config/<robot>/event_cfg.py`).
4. **Add observations** that read its position / velocity (mirror
   `tasks/dribbling/mdp/observations.py` for ball patterns).

---

## 3.14 Tweak experiment-level config (logger, seed, distributed)

These are top-level CLI-only changes:

| Change | Flag |
|---|---|
| Disable W&B | `--logger disabled` (subcommand picks `DISABLED` from `config/values/logger.py`) |
| Pick W&B project / run name | `--logger.project foo --logger.name bar` |
| Override learning steps for one run | `--learning-steps 5000000` (handled in `train.py`) |
| Resume from checkpoint | `--checkpoint <path>` |
| Warm start (load weights, reset step counter) | `--warm-start <path>` |
| Multi-GPU | `--cuda 0,1` (auto-relaunch under torchrun) |
| Custom seed | `--seed N` |
| Override env knobs at run time | `--task.env.scene.num-envs 2048` etc. |

Defaults live in `config/values/{algorithm,logger,task}.py`.

---

## 3.15 Add a new entry-point script

Goal: expose a new `pixi run` command that drives Colosseum.

1. **Write** the script under `src/colosseum/scripts/<name>.py`. Use
   `tyro.cli(<your-config-dataclass>)` to parse arguments and import
   `colosseum.tasks` to make sure registrations fire.
2. **Add an entry** in `pixi.toml` under `[tasks]`:
   ```toml
   my-cmd = { cmd = "python -m colosseum.scripts.my_name", description = "..." }
   ```
3. **Run** with `pixi run my-cmd`.

---

## 3.16 Modify the model registry

The CLI is `pixi run model-registry …`. Programmatically:

```python
from colosseum.utils.model_registry import ModelRegistry

ModelRegistry.register(task="t1-velocity", name="v2",
                       file="v2/t1-velocity_ppo_v2.onnx",
                       run="t1-velocity_PPO_42_<timestamp>", step=70_000_000)
ModelRegistry.set_default("t1-velocity", "v2")
ModelRegistry.remove("t1-velocity", "v0", delete_files=True)
```

The on-disk format is documented in the docstring of
`src/colosseum/utils/model_registry.py`.

---

## 3.17 Recap: which file changes for each feature?

| Feature | Mandatory file(s) |
|---|---|
| Task | `tasks/<task>/{__init__, config/__init__, config/<robot>/__init__, config/<robot>/t1_<task>_cfg, …}.py` |
| Algorithm impl | `algorithm/<name>.py` + `config/types/algorithm.py` |
| Reward term | `mdp/rewards.py` (or task-local) + `tasks/<task>/config/<robot>/reward_cfg.py` |
| Observation term | `mdp/observations.py` (or task-local) + `tasks/<task>/config/<robot>/observation_cfg.py` |
| Event term | `mjlab.envs.mdp.events` (or task-local) + `tasks/<task>/config/<robot>/event_cfg.py` |
| Termination term | `mjlab.envs.mdp.terminations` (or task-local) + `tasks/<task>/config/<robot>/cact_cfg.py` |
| Curriculum term | `mjlab` or `tasks/<task>/mdp/curriculums.py` + `tasks/<task>/config/<robot>/cact_cfg.py` |
| Command term | `tasks/<task>/mdp/<cmd>.py` + `tasks/<task>/config/<robot>/cact_cfg.py` |
| RMA encoder | `research/<paper>/rma_terms.py` (+ encoder networks) + task `make_env_cfg` |
| Abstraction term | `mdp/abstraction/<topic>/<term>.py` + task `make_env_cfg` |
| Constraint | `mdp/constraints.py` + `tasks/<task>/config/<robot>/constraint_cfg.py` |
| Robot | `robots/<robot>/{constants, actuators, collisions, sensors}.py` + XML |
| Asset | `assets/<asset>/<asset>_spec.py` |
| Script | `scripts/<name>.py` + `pixi.toml` |

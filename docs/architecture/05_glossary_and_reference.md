# 5. Glossary and reference

A flat reference for navigating the codebase — names, paths, and the
"gotchas" you only learn after grepping the repo a few times.

---

## 5.1 Glossary

| Term | Meaning |
|---|---|
| **mjlab** | The upstream framework. Provides `ManagerBasedRlEnv`, all standard managers, the Scene/Entity layer, and a MuJoCo Warp wrapper. Installed via `pixi.toml` from the `neverorfrog/mjlab` GitHub repo. |
| **Term** | One declarative unit configured by a `*TermCfg` dataclass and built into a runtime object. Reward, observation, event, termination, command, curriculum, abstraction, RMA-encoder, and constraint each have their own term type. |
| **Manager** | The class that owns and orchestrates a homogeneous collection of terms. Subclass of `mjlab.managers.ManagerBase`. |
| **`SceneEntityCfg`** | Bridge between a manager term and a scene entity. Stores `(name, body_names / joint_names / geom_names / site_names)` patterns. `resolve()` converts the patterns to global MuJoCo index tensors at startup. |
| **`MjSpec`** | MuJoCo's procedural model representation. Loaded from XML, possibly mutated in Python (e.g. adding an actuator block or a head camera), then compiled into `mjModel`. |
| **Decimation** | Number of physics sub-steps per env-step. `env.step_dt = sim.timestep * decimation`. |
| **Phase 1 / Phase 2** | RMA training phases. Phase 1: privileged encoder + policy trained jointly on PPO loss. Phase 2: adaptation encoder trained via MSE regression against the frozen Phase 1 latent. |
| **CaT** | "Constraints as Terminations" (arXiv:2403.18765). A constraint violation is converted into a stochastic termination probability with Polyak-averaged normalisation. |
| **Abstraction** | A planner-derived guidance signal cached at the env level (cost/direction maps, sub-goals). |
| **Privileged obs** | An observation group that contains ground-truth information unavailable at deployment time. Always read by the critic and (optionally) by RMA encoders; never by the actor. |
| **DAgger imitation** | Annealed `MSE(student_mean, teacher_action)` term added to PPO loss. The teacher's obs is a contiguous suffix of the student's obs. |
| **Run dir** | `<log_dir>/<run_name>/` containing `checkpoints/`, `wandb/`, `loguru.log`, and `experiment_config.yaml`. |
| **Pixi** | Conda + PyPI environment manager (see `pixi.toml`). The `default` env has CUDA 12.0 and is required for training. |

---

## 5.2 File-tree map (only the parts that matter for extension)

```
colosseum/
├── pixi.toml                      ← env management, pixi run task list
├── pyproject.toml                 ← packaging, ruff config
├── mkdocs.yml                     ← documentation site
├── README.md
├── docs/                          ← MkDocs site
│   ├── colosseum/                 ← user-facing tutorials
│   ├── mjlab/                     ← architectural notes about mjlab
│   ├── research/                  ← paper write-ups
│   └── architecture/              ← THIS REPORT
├── models/                        ← exported ONNX models + registry.yaml
├── checkpoints/                   ← convenience symlinks (manual)
├── src/colosseum/
│   ├── __init__.py                ← package version
│   ├── algorithm/                 ← BaseAlgorithm, PPO, RmaPPO, DAgger variants
│   │   ├── networks/              ← PpoActor, PpoValueNet, PrivilegedEncoder, TeacherPolicy
│   │   └── utils/                 ← rollout_buffer, normalization
│   ├── assets/ball/               ← ball entity factory
│   ├── config/
│   │   ├── types/                 ← TaskConfig, AlgorithmConfig, *Config dataclasses, registries
│   │   └── values/                ← default DEFAULTS dicts (task / algo / logger)
│   ├── envs/                      ← ColosseumEnv (canonical) + legacy specialised envs
│   ├── managers/                  ← rma_manager, abstraction_manager, constraint_manager
│   ├── mdp/                       ← reusable term functions (obs/rewards/constraints/...)
│   │   └── abstraction/maze/      ← grid abstractions, planners, sokoban solver
│   ├── research/                  ← paper-specific extensions and pipelines
│   │   ├── dribbling/             ← rma_terms, encoders, scripts (Phase 2, pipeline, eval)
│   │   └── soccer_maze/           ← benchmark configs, eval, plotting
│   ├── robots/                    ← per-robot configs (T1 23 DOF, ant)
│   ├── scripts/                   ← train, play, export-onnx, registry CLI, etc.
│   ├── tasks/                     ← per-task packages (velocity, dribbling, maze, soccer_maze)
│   ├── tests/                     ← actuator and contact sensor tests
│   └── utils/                     ← checkpoint, model_registry, logger, isaaclab math, …
└── typings/                       ← MuJoCo type stubs for static checking
```

---

## 5.3 Registries and where they live

| Registry | Type | Defined in | Populated by |
|---|---|---|---|
| `_TASK_REGISTRY` | `dict[str, TaskConfig]` | `config/types/task.py` | `@register_task("…")` decorators in each task's `t1_<task>_cfg.py` |
| `_ALGORITHM_REGISTRY` | `dict[str, (impl_cls, cfg_cls)]` | `config/types/algorithm.py` | `@register_algorithm("…", config_class=…)` decorators |
| `colosseum.config.values.task.DEFAULTS` | live view of `_TASK_REGISTRY` | `config/values/task.py` | imports `colosseum.tasks` to fire `@register_task` |
| `colosseum.config.values.algorithm.DEFAULTS` | `dict[str, AlgorithmConfig]` | `config/values/algorithm.py` | manual entry (`{"ppo": PPO_DEFAULT}`) |
| `colosseum.config.values.logger.DEFAULTS` | `dict[str, LoggerConfig]` | `config/values/logger.py` | manual entry (`{"wandb": …, "disabled": …}`) |
| `models/registry.yaml` | YAML file | `models/registry.yaml` | `utils/model_registry.py` (CLI `pixi run model-registry`) |

---

## 5.4 Pixi task ↔ script ↔ purpose

The `[tasks]` section of `pixi.toml` is the canonical mapping. Repeated
here for quick reference:

| `pixi run …` | Script | Purpose |
|---|---|---|
| `train` | `colosseum.scripts.train` | Custom PPO training (default). |
| `train-rsl-rl` | `colosseum.scripts.train_rsl_rl` | Trainer using mjlab's RSL-RL runner. |
| `train-mjlab` | `mjlab.scripts.train` | Train an mjlab built-in task. |
| `train-phase2` | `research/dribbling/scripts/train_phase2.py` | RMA Phase 2 adaptation encoder training. |
| `pipeline-dribbling` | `research/dribbling/scripts/pipeline_dribbling.py` | Multi-stage Phase 1 + Phase 2 pipeline. |
| `eval-dribbling` | `research/dribbling/scripts/evaluate_dribbling.py` | Evaluation protocol from the paper. |
| `obstacle-ablation` | `research/dribbling/scripts/obstacle_ablation.py` | Ablation of obstacle presence. |
| `play` | `colosseum.scripts.play` | Run a checkpoint / ONNX in a viewer. |
| `export-onnx` | `colosseum.scripts.export_onnx` | Export an actor to ONNX + register. |
| `model-registry` | `colosseum.scripts.model_registry_cli` | CLI for `models/registry.yaml`. |
| `cleanup-run` | `colosseum.scripts.cleanup_run` | Delete a run from disk + W&B. |
| `docs` / `build-docs` | `mkdocs serve` / `mkdocs build --strict` | Documentation. |

---

## 5.5 Important conventions and "gotchas"

### Auto-import order

`colosseum.tasks.__init__` walks the package with `pkgutil.iter_modules`
and imports every sub-package (skipping `utils` and `mdp`). This is what
fires the `@register_task` decorators. **A task whose top-level module
raises `ImportError` is silently skipped** — wrap optional imports in
`try / except ImportError: pass` (every existing task does this).

### Special term names

- A termination term named `arrived_at_goal` is treated as the success
  signal by `BaseAlgorithm.update_episode_counts`. Use that exact name for
  rolling success tracking in W&B.
- Reward terms producing per-env scalars are accumulated into
  `Episode_Reward/<term_name>` automatically.
- Constraint terms produce two metrics: `Episode_Constraint_violation/<name>`
  (% of steps with `p > 0`) and `Episode_Constraint_probability/<name>`.

### Float vs bool `terminated`

When `ConstraintManager` is active, `terminated` becomes a *float* tensor
(the per-env termination probability). The PPO and RmaPPO collect loops
guard with `if terminated.is_floating_point(): …` — preserve that pattern
in any new algorithm.

### Privileged groups in the rollout buffer

`RmaPPO._build_rollout_buffer` calls
`obs_mgr.group_obs_dim[group][0]` for every group in
`rma_manager.privileged_group_names`. **Every group an RMA term reads must
exist in `observations` before `_build_rollout_buffer` runs.** Make sure
your privileged obs groups are declared in `observation_cfg.py` *before*
passing them to `RmaTermCfg.privileged_obs_group`.

### Renaming a command

Commands are referenced by string in many places (rewards, observations,
constraints, curriculum). If you rename `commands["twist"]`, grep:

```
grep -rn '"twist"' src/colosseum/tasks/<task>/
```

…and rename every occurrence.

### `play_env_cfg` differences

By convention `play_env_cfg` (returned by `<TaskConfig>.play_env_cfg`):

- Sets `episode_length_s = int(1e9)` so the policy runs forever.
- Disables actor-obs corruption (`observations["actor"].enable_corruption = False`).
- Pops `push_robot` from `events`.
- Often lowers `num_envs` to 1.
- Disables training-only curricula (`cfg.curriculum = {}`).

### `algo_cfg.target` matters during play and export

`scripts/play.py` and `scripts/export_onnx.py` import the algo class via
`importlib.import_module(target)`. If you change `target` in the config but
forget to update old YAML files used for resume/replay, those old runs
will fail to load. Keep the import path stable across versions.

### Distributed: per-rank seed and EGL device

In multi-GPU runs, each rank uses `seed = config.seed + rank` so rollouts
diverge naturally. Every rank also sets
`MUJOCO_EGL_DEVICE_ID = local_rank` so the MuJoCo offscreen renderer is on
the right GPU. Don't override these env vars by hand.

### CLAUDE.md drift

The repository's `CLAUDE.md` references:
- `src/colosseum/robots/booster_t1/` — actually moved to
  `src/colosseum/robots/t1_23dof/`.
- `src/colosseum/train/tasks/cartpole/` — never existed; the cartpole doc
  in `docs/colosseum/cartpole.md` is a tutorial, not shipped code.

When in doubt, trust the actual source tree, not `CLAUDE.md`.

---

## 5.6 Frequently used file-grep cheatsheet

```bash
# Where is term X registered?
grep -rn 'register_task' src/colosseum/tasks
grep -rn 'register_algorithm' src/colosseum/algorithm

# What tasks consume a given command?
grep -rn 'command_name="ball_vel"' src/colosseum

# What rewards reference a given function?
grep -rn 'feet_distance_penalty' src/colosseum/tasks

# Which sensors does the dribbling task use?
grep -n 'SENSOR' src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py

# Where does an algorithm's optimizer get built?
grep -n '_build_optimizers' src/colosseum/algorithm/*.py
```

---

## 5.7 Minimum viable train / play / export cycle

For sanity-checking the full pipeline in under five minutes:

```bash
# 1. Quick train
pixi run train task:t1-velocity \
    --task.env.scene.num-envs 256 \
    --task.algo-cfg.learning-steps 200000 \
    --logger disabled

# 2. Play (auto-finds latest run)
pixi run play task:t1-velocity --num-envs 1

# 3. Export ONNX and register as v_test
pixi run export-onnx task:t1-velocity --name v_test

# 4. Replay via ONNX
pixi run play task:t1-velocity --agent onnx --onnx v_test --num-envs 1

# 5. Inspect / clean up
pixi run model-registry list
pixi run cleanup-run <run_name>
```

If any of these fails the failure points to a specific layer:

| Failure stage | Likely culprit |
|---|---|
| `train` won't start | `@register_task` not firing, or `algo_cfg.target` import path wrong |
| Train crashes early | Missing privileged group for an RMA term, or unresolved `SceneEntityCfg` pattern |
| `play` can't find ckpt | `resolve_checkpoint` path order — pass `--run-name` explicitly |
| `export-onnx` fails | Actor input dim mismatch (forgot to widen for RMA latents) |
| `play --agent onnx` produces wrong actions | ONNX exported before RmaPPO widened the actor; re-export with the matching algo |

---

## 5.8 Where to look next

- `docs/colosseum/internals.md` — the (slightly older) internal docs that
  inspired this report; useful for intuition on `ColosseumEnv`.
- `docs/mjlab/{overview, scene_simulation, task_layer}.md` — read these to
  understand the underpinnings.
- `docs/research/{dribbling, soccer-maze}.md` — the published context for
  the heaviest examples.
- `src/colosseum/tasks/velocity/` — the cleanest, smallest task to clone.
- `src/colosseum/tasks/dribbling/` — the most complex task; uses every
  Colosseum-specific feature (RMA, CaT, custom commands, DAgger).

This report is intended to stay in `docs/architecture/` and be checked in
alongside the rest of the documentation. If you add or rename a major
component, update the matching section here so future readers don't fall
into the same `CLAUDE.md`-style drift.

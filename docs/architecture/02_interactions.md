# 2. How the pieces interact

This file traces every important data flow at runtime: how an entry-point
script builds a task and an algorithm, how `ColosseumEnv` orchestrates its
managers in a single step, how `RmaPPO` composes the actor input, how DAgger
wires a frozen teacher, and how checkpoints / ONNX flow through scripts.

Use this in conjunction with `01_components.md`, which describes *what* every
piece is, and `03_modify_each_part.md`, which describes *how to extend* every
piece.

---

## 2.1 Entry-point flow: `pixi run train task:t1-velocity`

The lifecycle of a single command, from CLI to running rollout, is the
quickest way to see how the components are wired:

```
1. pixi.toml             →  python -m colosseum.scripts.train task:t1-velocity
2. scripts/train.py
     a. tyro.cli(TrainConfig)              ← config/types/experiment.py
        Subcommand list = colosseum.config.values.task.DEFAULTS
        which is _TASK_REGISTRY built by @register_task in tasks/*/config/__init__.py
        (task auto-import is triggered by `import colosseum.tasks` at top of train.py)
     b. _parse_cuda_devices → optionally relaunch under torchrun
     c. _init_distributed (NCCL or single-process)
     d. config.task.algo_cfg                    ← e.g. booster_t1_ppo_cfg()
     e. config.task.train_env_cfg               ← booster_t1_velocity_env_cfg()
     f. setup_wandb / setup_loguru, run_dir
     g. set_seed(rank_seed); configure_torch_backends()
3. _make_env(env_cfg, device)
     → env_cfg.class_type == ColosseumEnv     (set at the bottom of colosseum_env.py)
     → ColosseumEnv(cfg=env_cfg, device=device)
        - ManagerBasedRlEnv.__init__ builds Scene, compiles MjModel, creates Simulation
        - load_managers() instantiates standard managers + Colosseum extras (1.4)
4. importlib.import_module(algo_cfg.target)    ← e.g. "colosseum.algorithm.ppo"
   algo_class = PPO                            ← from "colosseum.algorithm.ppo:PPO"
   algo = algo_class(config=algo_cfg, env=env, device, log_fn, log_interval)
5. algo.attach_metadata(...)
   algo.configure_checkpointing(ckpt_dir, save_interval)
   if --warm-start or --checkpoint: algo.load(...)
6. algo.train()                                ← see 2.4 below
```

The same dispatcher (`importlib.import_module(target.rsplit(":", 1)[0])`) is
used by `play.py` and `export_onnx.py`. The `target` field in every
`PpoConfig` / `RmaPPOConfig` / `DaggerPpoConfig` is therefore the **single
source of truth** for which algorithm class runs.

---

## 2.2 Env construction order (inside `ColosseumEnv`)

`ColosseumEnv.load_managers()` (in `src/colosseum/envs/colosseum_env.py`) is
the only override of mjlab's manager-loading hook. It runs **once**, in this
order:

```
1. if cfg.abstractions:       AbstractionManager(cfg.abstractions, env=self)
2. super().load_managers()    # mjlab default managers:
                              #   ActionManager, ObservationManager,
                              #   RewardManager, TerminationManager,
                              #   EventManager, CommandManager,
                              #   CurriculumManager
3. if cfg.encoders:            self.rma_manager = RmaManager(cfg.encoders, env=self)
4. if cfg.constraints:         self.constraint_manager = ConstraintManager(cfg.constraints, env=self)
```

Why this order matters:

- `AbstractionManager` is built **before** the standard managers because
  several command terms read from it during `_prepare_terms()`
  (e.g. `AbstractionVelocityCommand` queries `manager.get_term("grid")`).
- `RmaManager` is built **after** the observation manager so its terms can
  inspect `obs_mgr.group_obs_dim[group]` to size their privileged encoders.
- `ConstraintManager` is independent and last; it only needs the scene.

`SceneEntityCfg.resolve()` is called by each manager's `_prepare_terms()`
(via mjlab's `_resolve_common_term_cfg` helper), turning name patterns like
`body_names=".*_knee"` into integer index tensors that runtime functions
will index directly.

---

## 2.3 Per-step flow inside `ColosseumEnv.step(action)`

Source: `src/colosseum/envs/colosseum_env.py`. With every optional manager
present, one call to `env.step(action)` is:

```
ColosseumEnv.step(action)
 ├─ obs, rewards, terminated, truncated, extras = ManagerBasedRlEnv.step(action)
 │     ├─ ActionManager.process_action → physics rollout (decimation × dt)
 │     ├─ EventManager (interval / reset hooks may fire)
 │     ├─ TerminationManager.compute  → terminated, truncated
 │     ├─ RewardManager.compute       → rewards
 │     ├─ CommandManager.compute      → resampling
 │     ├─ CurriculumManager.step      → schedule changes
 │     ├─ self._reset_idx(done_envs)  → cf. 2.5 below
 │     └─ ObservationManager.compute  → fresh obs dict
 │
 ├─ if rma_manager:        rma_manager.update()
 │     for term: term.update()         # e.g. push the new depth frame
 │
 ├─ if constraint_manager: cstr_prob = constraint_manager.compute()
 │     rewards = clip(rewards * (1 − cstr_prob), 0)
 │     terminated = float(cstr_prob); terminated[hard_done] = 1.0
 │
 └─ if abstraction_manager: abstraction_manager.compute(dt=self.step_dt)
        for term: term._update_signals(all_envs)   # signals usable next step
```

Two important consequences:

1. The RMA encoder runs **after** the observation manager, so its newest
   internal buffer (e.g. depth frame) is aligned with the obs that PPO will
   normalize and consume on the next iteration.
2. The constraint manager **mutates** `rewards` and `terminated` so that PPO
   does not need to know constraints exist. The float `terminated` is then
   recognised by `RmaPPO._collect_rollout` (which uses `dones = (terminated |
   truncated).float()` after a guard for `is_floating_point`) and by `PPO._collect_rollout`
   (which clamps the OR with `torch.clamp(terminated + truncated.float(), 0, 1)`).

---

## 2.4 PPO training loop (and what RmaPPO / DaggerPPO change)

Source: `src/colosseum/algorithm/ppo.py`, `rma_ppo.py`, `dagger_ppo.py`.

The default `PPO.train()` is:

```
for iteration in 1..N:
    collection_phase:
        for _step in num_steps_per_env:
            actor_obs = get_actor_obs(obs)
            critic_obs = get_critic_obs(obs)
            norm_actor = actor_obs_normalizer(actor_obs)
            norm_critic = critic_obs_normalizer(critic_obs)
            # Hook for subclasses to extend the actor input
            actor_input = self._compose_actor_input(norm_actor, privileged_obs)
            actions, log_probs, mean, std = actor.act_with_log_prob(actor_input)
            values = value_net(norm_critic)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            # Optional timeout-bootstrap if not is_finite_horizon
            rollout_buffer.add(...)

        last_values = value_net(critic_obs_normalizer(current_critic_obs))
        rollout_buffer.compute_returns_and_advantages(GAE)

    learning_phase:
        for batch in rollout_buffer.mini_batch_generator(num_mini_batches, num_epochs):
            # Re-evaluate normalized obs and (optionally) re-encode privileged inputs
            new_log_probs, entropy = actor.evaluate(actor_input, batch.actions)
            new_values = value_net(critic_input)
            kl = analytical_normal_kl(...)
            if config.schedule == "adaptive": adapt LR
            surrogate = clipped_ppo_loss
            loss = surrogate + value_coef * value_loss − entropy_coef * entropy.mean()
            optimizer.zero_grad(); loss.backward(); clip_grad_norm_(); optimizer.step()
```

Subclass hooks:

| Hook | Default | RmaPPO override | DaggerPPO override |
|---|---|---|---|
| `_build_networks` | `PpoActor(actor_dim) + PpoValueNet(critic_dim)` | Widens actor to `actor_dim + rma.total_latent_dim`. | (uses parent) |
| `_build_optimizers` | Joint `Adam(actor + value_net)` | Joint `Adam(actor + value_net + rma.parameters())` | (uses parent) |
| `_build_rollout_buffer` | Plain RolloutBuffer | Adds `privileged_obs_dims` for each privileged group. | Adds `extras={"teacher_actions": A}`. |
| `_compose_actor_input` | identity | `cat([norm_actor_obs, rma.encode(priv_obs, phase=p)])` | identity. |
| `get_privileged_obs(obs)` | `{}` | Pull `rma_manager.privileged_group_names` out of the obs dict. | (uses parent) |
| `_collect_rollout` | standard | Same shape, also stores privileged groups. | Stores `teacher.get_actions(actor_obs)`. |
| `_learning_step` | standard | Re-encodes privileged groups so encoder grads flow. | Adds `λ · MSE(student_mean, teacher_action)`, anneals `λ` linearly to 0. |

`RmaPPO._train_phase2()` is a separate loop (called when `self._phase == 2`)
that:

1. Freezes `actor`, `value_net`, and `rma_manager.privileged_parameters()`.
2. Builds a new optimizer over `rma_manager.adaptation_parameters()`.
3. Collects aligned temporal sequences (depth frames + privileged GT) for
   per-term sequence-MSE regression with TBPTT chunking and warmup.

The `train_phase2.py` script in `research/dribbling/scripts/` is what
normally invokes this branch.

---

## 2.5 Reset flow (`_reset_idx`)

`ColosseumEnv._reset_idx(env_ids)` is called from inside
`ManagerBasedRlEnv.step` whenever any environment is done:

```
1. ep_lens = episode_length_buf[env_ids].clone()       # snapshot before super zeroes them
2. super()._reset_idx(env_ids)
     ├─ EventManager runs "reset" mode events:
     │      reset_root_state_uniform, reset_joints_by_offset, reset_ball, ...
     ├─ TerminationManager / Curriculum / Command resets
     └─ ObservationManager.compute → fresh obs (consumed downstream)
3. abstraction_manager.reset(env_ids)                 → returns {} log dict
4. constraint_manager.reset(env_ids, ep_lens)         → logs Episode_Constraint_violation/<name>
5. rma_manager.reset(env_ids)                         → terms zero their internal state
   (e.g. DribblingRmaTerm zeros GRU hidden, depth frame, FOV mask)
```

The constraint manager's `reset` consumes `ep_lens` rather than re-reading
`episode_length_buf` because mjlab has already zeroed the buffer for the
done envs. The snapshot is what makes "% of steps violated" accurate.

---

## 2.6 RMA Phase 1 vs Phase 2 actor-input composition

In Phase 1 (`RmaPPO._compose_actor_input`):

```
priv_obs_dict = {
    "privileged_ball":      tensor[N, 4],
    "privileged_obstacles": tensor[N, 4],
}
z_phase1 = rma_manager.encode(priv_obs_dict, phase=1)      # concat per-term latents
        + N(0, latent_noise_std)                           # only when training
actor_input = cat([norm_proprio, z_phase1])                # [N, proprio + total_latent_dim]
```

In Phase 2 (during `_train_phase2`):

```
adapt_obs_dict = {"depth_frames": tensor[N, T, 1, H, W]}
z_phase2 = rma_manager.encode_phase2_with_fallback(priv_obs_dict, adapt_obs_dict)
        # falls back to detached Phase 1 latent for envs with `mask == False`
loss = sum(rma_manager.compute_adaptation_loss(priv_obs_dict, adapt_obs_dict, mask))
       # fallback loss is per-term latent MSE; concrete terms can override
       # with an auxiliary head loss (see DribblingRmaTerm.compute_loss).
```

`get_adaptation_mask()` is what makes Phase 2 robust to per-env signal drop
(e.g. ball outside camera FOV): for those envs we use the privileged
encoder detached so the policy is still well-conditioned during collection.

---

## 2.7 Action ↔ obs ↔ command interaction (the canonical example)

A typical step in the velocity task looks like this:

```
ObservationManager.compute(obs_dict)
   actor:   [base_ang_vel, projected_gravity, joint_pos, joint_vel,
             last_action, command(twist)]                    ← mjlab CommandManager
   critic:  actor + base_lin_vel + foot_height + foot_air_time + ...

PpoActor(actor)            → action_means, action_stds
JointPositionActionCfg     → q_target = q_default + scale * action
mjlab Action manager       → write to MuJoCo ctrl, run physics for `decimation` steps

RewardManager.compute(rewards)
   uses CommandManager.get_command("twist") to compute track_*_velocity etc.

CurriculumManager.step()
   commands_vel: at step thresholds, widens twist.ranges from a slow to fast schedule
```

The `command_name` parameter is the link: `cact_cfg.commands["twist"]`
exposes `"twist"`, every reward and observation that needs the velocity
command receives `"twist"` via `params={"command_name": "twist"}`. Renaming
the command means renaming everywhere — `commands["twist"]`,
`reward_cfg`'s `track_linear_velocity` params, observation
`generated_commands` params, and the curriculum.

---

## 2.8 Constraint Manager interaction with PPO

The CaT manager is invisible to the algorithm: PPO never imports
`ConstraintManager`. It only sees:

- A `terminated` tensor that may be float instead of bool.
- A `rewards` tensor already scaled by `(1 - p)`.

That is sufficient for the standard PPO loss and for GAE; the fact that the
manager is Polyak-averaging running maxes per term and producing a
stochastic termination mask is fully encapsulated. This is what lets you
add or remove constraint terms without retraining infrastructure code.

---

## 2.9 Abstraction Manager interaction with commands

The abstraction manager owns *terms*, but the coupling to the rest of the
MDP is through **other** terms (commands, rewards, observations) calling
`env.abstraction_manager.get_term("grid")` and using its task-specific
methods. For example, in the maze task:

- `AbstractionVelocityCommand` (`mdp/abstraction/maze/abstraction_velocity_command.py`):
  on `_resample`, queries `grid_term` for the direction to the next waypoint
  and turns it into a body-frame target velocity.
- `MazeGoalCommand` (`mdp/abstraction/maze/goal_command.py`): samples valid
  goal cells from the maze, exposes them as a `goal` command consumed by
  observations such as `agent_to_goal_vector`.

Because `_update_settings` is called on reset (`reset(env_ids)`), changing
the goal triggers a per-env rebuild of the cost/direction map for that env.
The `_maybe_rebuild` step uses change-detection on `AbstractionSettings` so
two envs with the same goal cell and identical (object-identity) maze map
share a cached solution.

---

## 2.10 Checkpoint and ONNX flow

```
algo.save(path, global_step=...) ─┐
                                   ├─ stores actor + value + optimizer
                                   │  + obs normalizers + config + metadata
                                   │  + (RmaPPO) rma_manager.state_dict()
                                   │  + (DaggerPPO) teacher metadata reference
algo.load(path) ──────────────────┘
                                   restore everything; PPO sets self.global_step

scripts/export_onnx.py
   resolve_checkpoint(...)
   ↓
   utils.export.export_policy_to_onnx
   ↓
   algo_class(config, env)          # builds matching networks
   algo.load(ckpt)
   algo.export_onnx(out_path)       # default impl in BaseAlgorithm
   ↓
   ModelRegistry.register(task, name, file, run, step)
   ↓
   models/<task>/<name>/<file>.onnx + symlink models/<task>/default.onnx

scripts/play.py --agent onnx
   _resolve_onnx(name | path | "default")
   ↓
   onnxruntime.InferenceSession
   ↓
   wraps it as `agent(obs_dict)` callable
```

The actor input layout *must* match between train and ONNX export — this is
why `play.py` re-instantiates the same `algo_class` (so PPO and RmaPPO build
identically-sized networks before loading the checkpoint).

---

## 2.11 Distributed training (torchrun)

`scripts/train.py` is "single-script multi-mode":

- `--cuda 0` → single GPU run.
- `--cuda 0,1` → relaunches itself under `torch.distributed.run` with one
  process per GPU.
- Inside the process: `_init_distributed` reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK`
  from the env, calls `torch.cuda.set_device(local_rank)`, sets
  `MUJOCO_EGL_DEVICE_ID`, and initializes a `nccl` process group.
- `BaseAlgorithm` exposes `_distributed_average_optimizer_grads` and
  `_distributed_sum_vector` / `_distributed_mean_scalar` that
  `PPO._learning_step` calls so gradients and metrics are synced across
  ranks; only rank 0 logs / saves.

Each rank seeds with `config.seed + rank` to diversify rollouts.

---

## 2.12 Summary diagram of the call graph

```
TrainConfig (tyro/yaml)
   └── task: TaskConfig  (registered subclass via @register_task)
          ├── train_env_cfg → ColosseumEnvCfg
          │        ├── scene (Scene/Entity configs from robots/, assets/)
          │        ├── observations / actions / rewards / events / commands /
          │        │   curriculum / terminations  (mjlab managers)
          │        └── encoders / abstractions / constraints  (Colosseum extras)
          └── algo_cfg → PpoConfig / RmaPPOConfig / DaggerPpoConfig (target=...)

scripts/train.py
   ├── ColosseumEnv(env_cfg)  ─── load_managers wires every manager
   ├── algo_class(algo_cfg, env)  ─── networks sized from env spaces
   └── algo.train()            ─── per-iteration rollout + update

ColosseumEnv.step(action)
   ├── ManagerBasedRlEnv.step (mjlab core)
   ├── rma_manager.update()
   ├── constraint_manager.compute()  → scale rewards / set float terminated
   └── abstraction_manager.compute(dt) → update guidance signals for next step
```

This is the only diagram you need to keep in your head — everything else in
the codebase is a leaf hanging off it.

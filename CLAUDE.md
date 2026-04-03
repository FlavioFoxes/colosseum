# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Colosseum** is a learning playground for humanoid robot research focused on robot soccer. It builds on top of **mjlab**, a GPU-accelerated reinforcement learning framework for robotics built on MuJoCo Warp.

Key characteristics:
- GPU-accelerated parallel training (4096+ environments)
- Manager-based declarative configuration (rewards, observations, actions)
- MuJoCo-native physics simulation
- Primary robot: Booster T1 humanoid (12 DOF locomotion / 23 DOF full body)

## Development Setup

### Environment Management

This project uses **Pixi** (a conda/pypi workspace manager) for environment and dependency management:

```bash
# Install dependencies and activate environment
pixi install

# Serve documentation locally
pixi run docs

# Build documentation
pixi run build-docs
```

The project uses:
- Python 3.12 (strict version)
- NVIDIA GPU required for training (MuJoCo Warp)
- `MUJOCO_GL=osmesa` environment variable (set automatically by Pixi)

**IMPORTANT: Always use `pixi run python` for Python commands:**
```bash
# Correct
pixi run python script.py

# Incorrect
python script.py  # Will fail with ModuleNotFoundError
```

This ensures the correct environment with all dependencies (mjlab, booster_robotics_sdk, etc.) is active.

### Code Quality

Ruff is configured with:
- Source directory: `src`
- Indent width: 2 spaces

## Architecture

### Three-Layer Structure

Colosseum follows mjlab's three-layer architecture:

1. **Simulation Layer** (lowest): GPU-accelerated physics via MuJoCo Warp
   - Batched parallel execution across thousands of environments
   - Direct `mjModel`/`mjData` access

2. **Scene Layer** (middle): Physical object organization
   - Entities (robots, terrain, obstacles)
   - Name → index mapping and resolution
   - Entity data access (poses, velocities, forces)

3. **Task Layer** (highest): Reinforcement learning MDP definition
   - Manager-based orchestration (Actions, Observations, Rewards, Terminations, Events)
   - Declarative term configuration
   - `gym.Env` interface for RL training

### Term-Based Pattern

The core design pattern across all layers:

1. **Declaration** (config time): Define terms declaratively using `*TermCfg` classes
2. **Resolution** (initialization): Map entity/component names → MuJoCo indices once
3. **Execution** (runtime): Fast computation using pre-resolved indices

Example term configuration:
```python
rewards = {
    "stand_reward": RewardTermCfg(
        func=mdp.body_height_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")}
    )
}
```

During initialization, `SceneEntityCfg.resolve()` converts `body_names="torso"` → `body_ids=[3]` (global MuJoCo index). At runtime, term functions use these pre-resolved indices for efficient tensor operations.

### Three-Layer MDP Architecture (Training/Deployment Sharing)

Colosseum implements a novel three-layer architecture for MDP functions (observations, rewards, etc.) that eliminates code duplication between training and deployment:

**Layer 1: Universal MDP Functions** (`src/colosseum/mdp/`)
- Robot-agnostic, physics-based pure functions
- Examples: `compute_projected_gravity()`, `exponential_reward_kernel()`
- Dependencies: Only torch and math utilities
- Used by: All robots and tasks, both training and deployment

**Layer 2: Robot-Specific MDP Functions** (`src/colosseum/robots/*/mdp/`)
- Functions specific to one robot platform
- Examples: `compute_foot_contact_state()` (T1-specific sensor processing)
- Dependencies: Layer 1 functions, robot constants
- Used by: Multiple tasks using the same robot

**Layer 3: Task-Specific MDP Functions** (`src/colosseum/tasks/*/mdp/`)
- Functions specific to one task, split into two modules:
  - `observations.py`: **Pure functions** (shared between training and deployment)
  - `wrappers.py`: **Training wrappers** (mjlab interface adapters, training-only)
- Examples: `compute_velocity_commands()`, `compute_joint_pos_rel()`
- Dependencies: Layer 1 and Layer 2 functions
- Used by: Training configs (via wrappers) AND deployment policies (via pure functions)

**Key Benefits:**
- ✅ **Single Source of Truth**: Observation logic defined once in pure functions
- ✅ **Guaranteed Consistency**: Training and deployment use identical computation
- ✅ **Easy Maintenance**: Change observation → works everywhere automatically
- ✅ **Testable**: Pure functions are independently unit-testable
- ✅ **Clear Separation**: Computation (pure functions) vs. integration (wrappers)

**Example Flow:**
```python
# Layer 1: Universal
def compute_projected_gravity(quat: Tensor) -> Tensor:
    # Generic quaternion rotation (works for any robot)
    ...

# Layer 3: Task-specific pure function
def compute_base_ang_vel(robot_data) -> Tensor:
    return robot_data.root_ang_vel_b  # Works in training AND deployment

# Layer 3: Training wrapper (training-only)
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    robot = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)  # Calls pure function

# Deployment: Direct use of pure function
class VelocityPolicy(Policy):
    def compute_observation(self):
        obs = compute_base_ang_vel(self.robot.data)  # Same pure function!
```

See `docs/SHARED_OBSERVATIONS.md` for detailed implementation guide.

## Directory Structure

```
src/colosseum/
├── mdp/              # Layer 1: Universal MDP functions (robot-agnostic)
│   ├── observations.py  # Pure observation functions (e.g., projected_gravity)
│   └── rewards.py       # Pure reward functions (e.g., exponential_kernel)
│
├── robots/           # Robot definitions and constants
│   ├── booster_t1/   # Booster T1 humanoid
│   │   ├── xmls/     # MuJoCo XML models
│   │   │   ├── T1_12dof.xml   # 12-DOF locomotion model (DEPRECATED)
│   │   │   └── T1_23dof.xml   # 23-DOF full body model (unified)
│   │   ├── t1_actuators.py    # Motor specs and actuator configs
│   │   ├── t1_contacts.py     # Collision and contact sensor configs
│   │   ├── t1_constants.py    # Spec loaders and entity configs (training)
│   │   ├── mdp/               # Layer 2: T1-specific MDP functions
│   │   │   └── observations.py  # T1-specific observations (e.g., foot_contact)
│   └── cartpole/     # CartPole balancing demo
│
├── tasks/            # Task definitions (training only)
│   ├── cartpole/     # CartPole balancing task
│   │   ├── cartpole_scene.py  # Scene configuration
│   │   ├── cartpole_task.py   # MDP (actions, obs, rewards, terminations, events)
│   │   └── mdp_functions.py   # Custom term functions
│   └── velocity/     # Velocity tracking task
│       ├── config/   # Training configurations (robot-specific)
│       │   └── t1/
│       │       └── env_cfgs.py  # T1 velocity env config
│       ├── mdp/      # Layer 3: Task-specific MDP functions
│       │   ├── observations.py  # Pure observation functions (SHARED)
│       │   └── wrappers.py      # Training wrappers (mjlab-only)
│       └── rl/       # RL algorithm configs
│
├── play/             # Evaluation/playback utilities
└── utils/            # Shared utilities (path helpers)

external/             # Git submodules (not managed by colosseum)
├── mjlab/            # Core RL framework
└── holosoma/         # Motion retargeting (optional)

tests/                # Test files
docs/                 # MkDocs documentation
```

## Creating Tasks

Tasks are registered via the `mjlab.tasks` entry point in `pyproject.toml`:

```toml
[project.entry-points."mjlab.tasks"]
cartpole = "colosseum.tasks.cartpole"
```

Each task module must:
1. Define scene configuration (`SceneCfg`)
2. Define MDP components (actions, observations, rewards, terminations, events)
3. Create `ManagerBasedRlEnvCfg` combining all components
4. Call `register_mjlab_task()` with a unique `task_id`

See `src/colosseum/train/tasks/cartpole/` for a complete minimal example.

## Working with Robots

### Booster T1 Humanoid

Located in `src/colosseum/robots/booster_t1/`:

#### XML Models

- **12 DOF model** ([T1_12dof.xml](src/colosseum/robots/booster_t1/xmls/T1_12dof.xml)): Legs only (6 DOF per leg), used for locomotion training
- **23 DOF model** ([T1_23dof.xml](src/colosseum/robots/booster_t1/xmls/T1_23dof.xml)): Full body including arms, waist, and neck, for deployment

#### Configuration Modules

The T1 configuration is organized into three modules:

**[t1_actuators.py](src/colosseum/robots/booster_t1/t1_actuators.py)**: Motor specifications and actuator configurations
- `MOTOR_SPECS`: Dictionary of motor specifications from manufacturer data (gear ratio, torque, speed, inertia)
- `compute_pd_gains()`: Computes PD controller gains using Unitree G1 method (natural frequency + damping ratio)
- Actuator configs for 12-DOF locomotion:
  - `T1_ACTUATOR_HIP_PITCH`, `T1_ACTUATOR_HIP_ROLL`, `T1_ACTUATOR_HIP_YAW`
  - `T1_ACTUATOR_KNEE`
  - `T1_ACTUATOR_ANKLE_PITCH`, `T1_ACTUATOR_ANKLE_ROLL`
- Actuator configs for 23-DOF full body:
  - `T1_ACTUATOR_NECK`, `T1_ACTUATOR_ARM`, `T1_ACTUATOR_WAIST`

**[t1_contacts.py](src/colosseum/robots/booster_t1/t1_contacts.py)**: Collision and contact sensor configurations
- Collision configs (modify geom properties):
  - `FEET_ONLY_COLLISION`: Only foot geoms collide (recommended for training)
  - `FULL_COLLISION_WITHOUT_SELF`: All parts collide with environment, no self-collision
  - `FULL_COLLISION`: Full collision including self-collision (most realistic)
  - `HANDS_FEET_COLLISION`: Only hands and feet collide (for manipulation)
- Contact sensor configs (for observation/reward):
  - `FEET_GROUND_CONTACT_SENSOR`: Tracks foot-ground contact with air time
  - `SELF_COLLISION_SENSOR`: Detects self-collisions
  - `HAND_CONTACT_SENSOR`: Tracks hand contact for manipulation
- `T1_FOOT_GEOM_NAMES`: Tuple of all foot geometry names for events

**[t1_constants.py](src/colosseum/robots/booster_t1/t1_constants.py)**: Spec loaders and entity configurations
- XML paths: `T1_12DOF_XML`, `T1_23DOF_XML`
- Spec loaders:
  - `get_t1_12dof_spec()`: Returns MjSpec for 12-DOF locomotion model
  - `get_t1_23dof_spec()`: Returns MjSpec for 23-DOF full body model
- Pre-configured entity configs:
  - `T1_12DOF_ENTITY_CFG`: Complete entity config for locomotion training
  - `T1_23DOF_ENTITY_CFG`: Complete entity config for full body deployment

#### PD Gain Computation

T1 uses the Unitree G1 method for computing actuator gains:

```python
# Natural frequency and damping ratio (same as G1)
natural_freq = 10.0 * 2π  # 10Hz in rad/s
damping_ratio = 2.0  # Overdamped (prevents oscillations)

# Compute gains from motor reflected inertia
stiffness = reflected_inertia × ω_n²
damping = 2 × ζ × reflected_inertia × ω_n
```

This ensures stable, overdamped control appropriate for each joint's mechanical properties.

### Entity Configuration

#### Using Pre-Configured Entities

The simplest way to use T1 is with the pre-configured entity configs:

```python
from colosseum.robots.booster_t1.t1_constants import T1_12DOF_ENTITY_CFG

# Use directly in scene config
entities = {"robot": T1_12DOF_ENTITY_CFG}
```

These configs include all actuators, collision settings, and sensors with manufacturer-accurate specifications.

#### Custom Entity Configuration

Entities are configured using `EntityCfg`:
- `spec_fn`: Function returning `mujoco.MjSpec` (e.g., `get_t1_12dof_spec`)
- `init_state`: Initial joint positions/velocities
- `actuators`: Dict of actuator configs applying to joint patterns
- `articulation_info`: Optional collision/armature settings using `EntityArticulationInfoCfg`:
  - `collision`: `CollisionCfg` object (e.g., `FEET_ONLY_COLLISION`)
  - `armature`: Joint armature values (reflected inertia)
- `sensors`: List of sensor configs (e.g., `[FEET_GROUND_CONTACT_SENSOR]`)

## Documentation

Documentation is built with MkDocs Material and includes:
- Colosseum-specific guides (Getting Started, CartPole tutorial)
- mjlab architecture documentation (3-layer system, term-based patterns)

The documentation covers:
- **Overview**: Architecture and design patterns
- **Task Layer**: Manager-based RL environment, term configuration
- **Scene Layer**: Entities, compilation, indexing
- **Simulation Layer**: MuJoCo Warp integration

## Key Concepts

### SceneEntityCfg Resolution

`SceneEntityCfg` is the bridge between managers and entities:
- Stores entity name + component patterns (e.g., `body_names=".*_knee"`)
- During `resolve()`, converts patterns to global MuJoCo indices
- Passed to term functions at runtime with pre-resolved indices

### MuJoCo Compilation Flow

1. Create `Scene` with entity configs
2. Each entity loads its `MjSpec` (XML representation)
3. Scene merges all specs and compiles → `MjModel` (MuJoCo assigns global indices)
4. Initialize entities with compiled model → creates `EntityIndexing`
5. Resolve `SceneEntityCfg` objects to map names → indices

### Custom Term Functions

Term functions follow this signature:
```python
def my_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    data = entity.data.body_pos_w[:, asset_cfg.body_ids]  # Pre-resolved indices
    return compute_reward(data)
```

## Deployment System Architecture

Deployment is handled entirely by the **C++ arena runtime** at `src/arena/`. The Python `src/colosseum/` tree is training infrastructure only — no deployment code lives there.

### Directory Structure

```
src/arena/
├── include/
│   ├── TaskConfig.h          # Top-level config (task name, model path, policy_dt, action_scale, RobotConfig)
│   ├── RobotConfig.h         # RobotConfig<N>: joint names, gains, limits, armature, prepare_state
│   ├── RobotData.h           # RobotData<N>: joint/IMU state + precomputed real2sim/sim2real maps
│   ├── RobotState.h          # Pure sensor snapshot written by portals (hardware order)
│   ├── Policy.h              # Abstract base: owns ONNX model, input_source_, robot_data_
│   ├── ObservationSpec.h     # Named component list + validate_size() for debug builds
│   ├── VelocityCommand.h     # VelocityCommand + set_normalized(); owned by concrete task
│   ├── IInputSource.h        # Generic get_axis(int)/get_button(int); factory create_input_source()
│   ├── TaskRegistry.h        # REGISTER_TASK macro + singleton registry
│   ├── ModelRegistry.h       # Resolves task name → ONNX model path
│   ├── OnnxPolicy.h          # ONNX Runtime wrapper (infer, input_dim, output_dim)
│   ├── input/
│   │   ├── JoystickInput.h   # evdev polling thread, per-axis atomics, deadzone
│   │   └── KeyboardInput.h   # WASD/QE raw terminal input
│   └── portals/
│       ├── IPortal.h         # initialize / hasState / getState / updateState / publishCommand / tick
│       ├── MujocoPortal.h    # GLFW sim-to-sim backend
│       ├── RobotPortal.h     # Booster SDK real-robot backend
│       └── CircusPortal.h    # TCP/msgpack sim backend
│
├── src/
│   ├── main.cpp              # CLI entry point: --backend booster|mujoco|circus --task <name>
│   ├── tasks/
│   │   └── T1VelocityFlat.cpp  # T1 23-DOF velocity task (obs, config, REGISTER_TASK)
│   ├── portals/
│   │   ├── MujocoPortal.cpp
│   │   ├── RobotPortal.cpp
│   │   └── CircusPortal.cpp
│   └── input/
│       ├── JoystickInput.cpp
│       └── KeyboardInput.cpp
│
└── assets/
    └── scene.xml             # MuJoCo scene (ground, lights) — robot MJCF composed at runtime
```

### Configuration

All task configuration is set in the task's static `make_config()` method (e.g. `T1VelocityFlat::make_config()`). There are no external config files.

**TaskConfig** (top-level):
```cpp
struct TaskConfig {
    static constexpr int NUM_JOINTS = 23;
    std::string task_name;       // registry key, also used for model lookup
    std::string model_path;      // resolved ONNX path
    float policy_dt = 0.02f;     // 50 Hz
    float action_scale = 0.25f;  // scalar; per-joint scale = action_scale * effort_limit[i] / kp[i]
    RobotConfig<23> robot;       // all hardware specs
    std::string scene_mjcf_path; // MujocoPortal scene (ground, lights)
};
```

**RobotConfig<N>** (hardware specs):
```cpp
// joint_names       = hardware/DDS order (what the robot reports)
// sim_joint_names   = MuJoCo compiled order (what the policy was trained with)
// For T1 these are identical → real2sim/sim2real are identity permutations.
cfg.robot.joint_stiffness   // Kp per joint — MUST match training
cfg.robot.joint_damping     // Kd per joint — MUST match training
cfg.robot.effort_limit      // peak torque per joint [Nm]
cfg.robot.joint_armature    // reflected inertia (0.3 for all T1 joints)
cfg.robot.default_joint_pos // standing pose; added to decoded action
cfg.robot.prepare_state     // safe startup pose (pos, stiffness, damping, duration_s)
cfg.robot.mjcf_path         // robot MJCF — composed with scene at runtime by MujocoPortal
```

### Joint Order Remapping

`RobotData<N>` (owned by `Policy` base) precomputes two index arrays at construction:

- `real2sim[i]` — hardware index that feeds sim slot `i`. Used in `build_observation()`:
  `obs_joint_pos[i] = state.joint_pos[real2sim[i]]`
- `sim2real[i]` — sim index that feeds hardware slot `i`. Used in `get_action()`:
  `hardware_target[i] = net_out[sim2real[i]] * scale + default_joint_pos[i]`

`last_action` (fed back into the observation) is kept in sim/policy order — it is the raw network output, not remapped.

### Control Loop

```
main.cpp:
  portal->initialize()
  [booster only] portal->changeMode(kCustom) → portal->prepare(cfg.robot.prepare_state)
  policy->reset()
  loop:
    portal->updateState()
    targets = policy->get_action(portal->getState())
    portal->publishCommand(targets, kp, kd)
    portal->tick()   ← sleeps to maintain policy_dt
```

### Adding a New Task

1. Create `src/arena/src/tasks/MyTask.cpp`.
2. Define a class inheriting `Policy`. Implement `build_observation()` and `make_config()`.
3. Call `REGISTER_TASK("my-task", MyTask);` at file scope.
4. Add a model file under `src/arena/models/` and register the name in `ModelRegistry`.

The task is automatically available via `--task my-task` at runtime.

### Backends

| Flag | Portal | Notes |
|------|--------|-------|
| `--backend booster` | `RobotPortal` | Real Booster T1 via DDS/SDK. Runs `prepare()` before loop. |
| `--backend mujoco` | `MujocoPortal` | GLFW viewer. Composes scene+robot MJCF at runtime. Sets armature. |
| `--backend circus` | `CircusPortal` | TCP/msgpack sim. Sends PD torques, receives state. |
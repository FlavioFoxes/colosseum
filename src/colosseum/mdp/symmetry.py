"""Layer 1: Universal symmetry utilities (robot-agnostic).

Provides:
- MirrorableObservationTermCfg: ObservationTermCfg subclass that carries a mirror_fn.
- Generic mirror functions for common observation types (ang_vel, gravity, command).
- TermMirrorSpec / build_symmetry_spec: compile an obs group's mirror layout once.
- mirror_obs: apply the compiled spec to a batched observation tensor.

Usage pattern
-------------
1. Declare mirror_fn on each obs term in observation_cfg.py:

    from colosseum.mdp.symmetry import MirrorableObservationTermCfg, mirror_ang_vel
    from colosseum.robots.t1_23dof.mdp.symmetry import mirror_joints

    "base_ang_vel": MirrorableObservationTermCfg(
        func=builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        mirror_fn=mirror_ang_vel,
    )
    "joint_pos": MirrorableObservationTermCfg(
        func=joint_pos_rel,
        mirror_fn=mirror_joints,
    )
    # Terms that need no mirroring just use mirror_fn=None (default).

2. In the algorithm, build the spec once from the env's observation manager:

    from colosseum.mdp.symmetry import build_symmetry_spec, mirror_obs

    self.actor_sym_spec = build_symmetry_spec(env.observation_manager, "actor")

3. Mirror a batch of observations:

    mirrored = mirror_obs(actor_obs_norm, self.actor_sym_spec)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from mjlab.managers.observation_manager import ObservationTermCfg


# ---------------------------------------------------------------------------
# Subclass: ObservationTermCfg + mirror_fn
# ---------------------------------------------------------------------------

@dataclass
class MirrorableObservationTermCfg(ObservationTermCfg):
  """ObservationTermCfg extended with an optional left-right mirror function.

  mirror_fn: pure function (Tensor) -> Tensor applied to one term's slice of
  the concatenated observation vector during symmetry-loss computation.
  None means the term is invariant under the mirror and the slice is copied as-is.
  """
  mirror_fn: Callable[[torch.Tensor], torch.Tensor] | None = None


# ---------------------------------------------------------------------------
# Generic mirror functions for common observation types
# ---------------------------------------------------------------------------

def mirror_ang_vel(x: torch.Tensor) -> torch.Tensor:
  """Mirror base angular velocity under left-right (y → -y) reflection.

  Angular velocity is a pseudovector. Under reflection P = diag(1,-1,1):
    ω' = det(P) * P · ω = -[ωx, -ωy, ωz] = [-ωx, ωy, -ωz]

  Args:
    x: (..., 3) tensor [wx, wy, wz].

  Returns:
    (..., 3) mirrored tensor [-wx, wy, -wz].
  """
  return x * x.new_tensor([-1.0, 1.0, -1.0])


def mirror_projected_gravity(x: torch.Tensor) -> torch.Tensor:
  """Mirror projected gravity vector under left-right (y → -y) reflection.

  Gravity is a regular vector in the body frame.
  Under y → -y: [gx, gy, gz] → [gx, -gy, gz].

  Args:
    x: (..., 3) tensor [gx, gy, gz].

  Returns:
    (..., 3) mirrored tensor [gx, -gy, gz].
  """
  return x * x.new_tensor([1.0, -1.0, 1.0])


def mirror_velocity_command(x: torch.Tensor) -> torch.Tensor:
  """Mirror velocity command under left-right (y → -y) reflection.

  [vx, vy, vyaw] → [vx, -vy, -vyaw]:
  - Forward speed vx is unchanged.
  - Lateral speed vy flips (positive = left → negative = right).
  - Yaw rate vyaw flips (positive = turn-left → negative = turn-right).

  Args:
    x: (..., 3) tensor [vx, vy, vyaw].

  Returns:
    (..., 3) mirrored tensor [vx, -vy, -vyaw].
  """
  return x * x.new_tensor([1.0, -1.0, -1.0])


def mirror_gait_phase(x: torch.Tensor) -> torch.Tensor:
  """Mirror gait phase clock under left-right reflection.

  [cos_L, cos_R, sin_L, sin_R] → [cos_R, cos_L, sin_R, sin_L]

  No sign changes: cosine and sine are evaluated on phases, not joint angles,
  and the reflection simply swaps which foot is "left" and which is "right".
  """
  return x[..., [1, 0, 3, 2]]


# ---------------------------------------------------------------------------
# Symmetry spec: compiled layout for efficient mirror_obs calls
# ---------------------------------------------------------------------------

@dataclass
class TermMirrorSpec:
  """Compiled slice spec for one observation term."""
  start: int
  end: int
  base_dim: int        # per-timestep dimension
  history_length: int  # 0 = no history
  mirror_fn: Callable[[torch.Tensor], torch.Tensor] | None


def build_symmetry_spec(obs_manager, group_name: str) -> list[TermMirrorSpec]:
  """Compile a symmetry spec from the observation manager for one group.

  Call once at algorithm initialisation; the returned spec is allocation-free
  to apply at training time via mirror_obs().

  Args:
    obs_manager: mjlab ObservationManager (env.observation_manager).
    group_name:  Observation group name, e.g. "actor".

  Returns:
    List of TermMirrorSpec, one per term in declaration order.
  """
  term_names = obs_manager.active_terms[group_name]
  term_dims = obs_manager.group_obs_term_dim[group_name]

  specs: list[TermMirrorSpec] = []
  offset = 0

  for name, dims in zip(term_names, term_dims):
    total_size = math.prod(dims)
    cfg = obs_manager.get_term_cfg(group_name, name)
    mirror_fn = getattr(cfg, "mirror_fn", None)
    history = getattr(cfg, "history_length", 0) or 0
    base_dim = total_size // history if history > 0 else total_size

    specs.append(TermMirrorSpec(
      start=offset,
      end=offset + total_size,
      base_dim=base_dim,
      history_length=history,
      mirror_fn=mirror_fn,
    ))
    offset += total_size

  return specs


def mirror_obs(obs: torch.Tensor, spec: list[TermMirrorSpec]) -> torch.Tensor:
  """Apply left-right mirror to a concatenated observation tensor.

  Terms with mirror_fn=None are copied unchanged (invariant under the mirror).
  Terms with history are reshaped to (..., H, D), mirrored per-timestep, then
  flattened back.

  Args:
    obs:  (..., obs_dim) observation tensor (already normalised or raw).
    spec: compiled spec from build_symmetry_spec().

  Returns:
    (..., obs_dim) mirrored tensor, same dtype and device as input.
  """
  chunks: list[torch.Tensor] = []

  for term in spec:
    chunk = obs[..., term.start:term.end]

    if term.mirror_fn is not None:
      if term.history_length > 0:
        # Shape: (..., H * D) → (..., H, D) → mirror each step → (..., H * D)
        leading = chunk.shape[:-1]
        chunk = chunk.view(*leading, term.history_length, term.base_dim)
        chunk = term.mirror_fn(chunk)
        chunk = chunk.view(*leading, -1)
      else:
        chunk = term.mirror_fn(chunk)

    chunks.append(chunk)

  return torch.cat(chunks, dim=-1)

"""Layer 2: T1-specific symmetry (left-right reflection about the sagittal plane).

Defines the joint permutation and sign-flip mask needed to mirror any joint-indexed
tensor (positions, velocities, actions) from the left side to the right side and
vice versa.

Anatomy reference (JOINT_NAMES ordering from t1_23dof/constants.py):
  0  Left_Hip_Pitch        6  Right_Hip_Pitch
  1  Left_Hip_Roll         7  Right_Hip_Roll
  2  Left_Hip_Yaw          8  Right_Hip_Yaw
  3  Left_Knee_Pitch       9  Right_Knee_Pitch
  4  Left_Ankle_Pitch     10  Right_Ankle_Pitch
  5  Left_Ankle_Roll      11  Right_Ankle_Roll
  12 Waist (yaw, z-axis)
  13 Left_Shoulder_Pitch  17  Right_Shoulder_Pitch
  14 Left_Shoulder_Roll   18  Right_Shoulder_Roll
  15 Left_Elbow_Pitch     19  Right_Elbow_Pitch
  16 Left_Elbow_Yaw       20  Right_Elbow_Yaw
  21 AAHead_yaw
  22 Head_pitch

Sign convention: reflection P maps y → -y (sagittal-plane mirror).
  - Pitch joints (axis ≈ y): sign = +1  (rotation direction unchanged)
  - Roll  joints (axis ≈ x): sign = -1  (pseudovector, det(P)=-1 flips sign)
  - Yaw   joints (axis ≈ z): sign = -1  (pseudovector, det(P)=-1 flips sign)
"""

import torch

# Permutation: index i holds the index in the original tensor that should go to
# position i after the left-right swap.
JOINT_MIRROR_INDICES: list[int] = [
  6, 7, 8, 9, 10, 11,   # R_Hip_{Pitch,Roll,Yaw}, R_Knee, R_Ankle_{Pitch,Roll}
  0, 1, 2, 3, 4, 5,     # L_Hip_{Pitch,Roll,Yaw}, L_Knee, L_Ankle_{Pitch,Roll}
  12,                    # Waist (singleton, stays)
  17, 18, 19, 20,        # R_Shoulder_{Pitch,Roll}, R_Elbow_{Pitch,Yaw}
  13, 14, 15, 16,        # L_Shoulder_{Pitch,Roll}, L_Elbow_{Pitch,Yaw}
  21,                    # AAHead_yaw (stays)
  22,                    # Head_pitch (stays)
]

# Sign mask: +1 for pitch joints (symmetric), -1 for roll/yaw joints (anti-symmetric).
# Applied AFTER the permutation (i.e. to the already-swapped tensor).
JOINT_MIRROR_SIGNS: list[float] = [
  +1, -1, -1, +1, +1, -1,  # R_Hip_{P,Ro,Y}, R_Knee, R_Ankle_{P,Ro}  (now in left slots)
  +1, -1, -1, +1, +1, -1,  # L_Hip_{P,Ro,Y}, L_Knee, L_Ankle_{P,Ro}  (now in right slots)
  -1,                       # Waist (yaw → anti-symmetric)
  +1, -1, +1, -1,           # R_Shoulder_{P,Ro}, R_Elbow_{P,Y}
  +1, -1, +1, -1,           # L_Shoulder_{P,Ro}, L_Elbow_{P,Y}
  -1,                       # AAHead_yaw (anti-symmetric)
  +1,                       # Head_pitch (symmetric)
]

# Pre-build as tensors so mirror_joints is allocation-free at runtime.
_MIRROR_IDX = torch.tensor(JOINT_MIRROR_INDICES, dtype=torch.long)
_MIRROR_SIGNS = torch.tensor(JOINT_MIRROR_SIGNS, dtype=torch.float32)


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
  """Mirror a joint-indexed tensor under left-right reflection.

  Works for joint positions, velocities, and actions (same 23-DOF ordering).

  Args:
    x: (..., 23) tensor in JOINT_NAMES order.

  Returns:
    (..., 23) mirrored tensor.
  """
  signs = _MIRROR_SIGNS.to(x.device)
  idx = _MIRROR_IDX.to(x.device)
  return x[..., idx] * signs

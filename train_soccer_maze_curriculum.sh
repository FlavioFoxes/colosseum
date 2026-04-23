#!/usr/bin/env bash
# Manual curriculum training for t1-soccer-maze.
#
# Phases:
#   0 — open 3×3 interior, no walls
#   1 — open 5×5 interior, no walls
#   2 — small maze 5×5 interior with walls
#   3 — medium maze 6×6 interior with walls
#   4 — large maze 10×10 interior with walls
#
# Each phase runs to completion, then its latest.pt is copied to a fixed path
# before the next phase starts.
#
# Usage:
#   bash train_soccer_maze_curriculum.sh              # run all phases from scratch
#   bash train_soccer_maze_curriculum.sh 2            # start from phase 2 (needs phase 1 checkpoint)
#   bash train_soccer_maze_curriculum.sh 2 3          # run only phases 2 and 3

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_ENVS=4000
CUDA="0"
LOG_DIR="./logs/soccer_maze_curriculum"
CKPT_DIR="./checkpoints/soccer_maze_curriculum"
WANDB_GROUP="soccer-maze-curriculum"
# ──────────────────────────────────────────────────────────────────────────────

START_PHASE=${1:-0}
END_PHASE=${2:-4}

mkdir -p "$CKPT_DIR"

BASE=(
  pixi run -e train train task:t1-soccer-maze
  --task.env.scene.num-envs "$NUM_ENVS"
  --cuda "$CUDA"
  logger:wandb
  --logger.log-dir "$LOG_DIR"
  --logger.group "$WANDB_GROUP"
)

find_latest_checkpoint() {
  # Returns the most recently modified latest.pt under LOG_DIR.
  find "$LOG_DIR" -name "latest.pt" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}'
}

run_phase() {
  local phase=$1
  local checkpoint=$2  # empty string = fresh start

  echo ""
  echo "════════════════════════════════════════════════"
  echo "  Phase ${phase}"
  echo "════════════════════════════════════════════════"

  local cmd=("${BASE[@]}" --maze-phase-index "$phase")
  [[ -n "$checkpoint" ]] && cmd+=(--checkpoint "$checkpoint")

  echo "Command: ${cmd[*]}"
  echo ""
  "${cmd[@]}"

  local latest
  latest=$(find_latest_checkpoint)
  if [[ -z "$latest" ]]; then
    echo "ERROR: could not find latest.pt after phase ${phase}" >&2
    exit 1
  fi

  cp "$latest" "${CKPT_DIR}/phase${phase}.pt"
  echo ""
  echo "Saved phase ${phase} checkpoint → ${CKPT_DIR}/phase${phase}.pt"
}

for phase in $(seq "$START_PHASE" "$END_PHASE"); do
  if [[ $phase -eq 0 ]]; then
    run_phase 0 ""
  else
    prev=$((phase - 1))
    prev_ckpt="${CKPT_DIR}/phase${prev}.pt"
    if [[ ! -f "$prev_ckpt" ]]; then
      echo "ERROR: checkpoint for phase ${prev} not found at ${prev_ckpt}" >&2
      echo "       Run phase ${prev} first, or place the checkpoint there manually." >&2
      exit 1
    fi
    run_phase "$phase" "$prev_ckpt"
  fi
done

echo ""
echo "Curriculum complete. Checkpoints saved in ${CKPT_DIR}/"

#!/usr/bin/env python3
"""Export a colosseum PPO checkpoint to ONNX.

Output layout:
    models/<task_name>/<task_name>_<timestamp>.onnx
    models/<task_name>/<task_name>_latest.onnx  (symlink → above)

Usage:
    pixi run -e train export-onnx task:t1-velocity-flat
    pixi run -e train export-onnx task:t1-velocity-rough --checkpoint ./logs/wandb/latest-run/checkpoints
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import tyro
from loguru import logger
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from colosseum.algorithm.base_algorithm import get_latest_checkpoint
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.train.export import export_policy_to_onnx


def _resolve_checkpoint(checkpoint: str | None) -> Path | None:
  """Resolve checkpoint path, supporting 'latest', directories, and direct paths."""
  if not checkpoint or checkpoint.lower() == "latest":
    ckpt_dir = Path("./logs/wandb/latest-run/checkpoints")
    if ckpt_dir.exists():
      return get_latest_checkpoint(ckpt_dir.resolve())
    return None

  p = Path(checkpoint)
  if p.is_dir():
    return get_latest_checkpoint(p.resolve())
  return p.resolve() if p.exists() else None


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ExportConfig(BaseExperimentConfig):
  """Export configuration."""

  checkpoint: str = "latest"


def main() -> None:
  """Export a trained policy checkpoint to ONNX."""
  config = tyro.cli(ExportConfig, config=(tyro.conf.CascadeSubcommandArgs,))

  ckpt = _resolve_checkpoint(config.checkpoint)
  if ckpt is None or not ckpt.exists():
    logger.error(f"No checkpoint found for: {config.checkpoint}")
    sys.exit(1)

  task_name = config.task.name
  task_models_dir = Path("models") / task_name
  task_models_dir.mkdir(parents=True, exist_ok=True)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"{task_name}_{timestamp}.onnx"
  output_path = task_models_dir / filename

  result = export_policy_to_onnx(config, ckpt, output_path)

  # Update stable latest symlink
  latest_link = task_models_dir / f"{task_name}_latest.onnx"
  try:
    if latest_link.exists() or latest_link.is_symlink():
      latest_link.unlink()
    latest_link.symlink_to(result.name)
    logger.info(f"Symlink: {latest_link.name} -> {result.name}")
  except OSError as e:
    logger.warning(f"Could not create latest symlink: {e}")

  logger.success(f"Exported: {result}")


if __name__ == "__main__":
  main()

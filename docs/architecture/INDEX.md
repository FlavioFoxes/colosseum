# Colosseum Architecture Report — Index

This report explains how the **Colosseum** repository (a humanoid-soccer
research playground built on top of [mjlab](https://github.com/neverorfrog/mjlab))
is structured, how the pieces interact, and how to extend each piece with new
content (tasks, algorithms, reward terms, encoders, abstractions, constraints,
robots, …).

Everything described here lives under `src/colosseum/` unless explicitly noted.
Path references in the body of the report are relative to the repository root.

The report is split into five topical files; this file is the entry point.

| # | File | What's inside |
|---|------|---------------|
| 1 | [01_components.md](01_components.md) | All the elements that compose the architecture (mjlab three layers, Colosseum-specific extensions, task package layout, algorithms, scripts, configs, model registry). |
| 2 | [02_interactions.md](02_interactions.md) | How the elements interact: env construction order, `step()` and `_reset_idx()` flow inside `ColosseumEnv`, RmaPPO's actor-input composition, training-loop contract between scripts, configs, algorithms and env. |
| 3 | [03_modify_each_part.md](03_modify_each_part.md) | Step-by-step recipes for changing each part: add a task, an algorithm, a reward / observation / event / termination / curriculum / command term, an RMA encoder, an abstraction term, a constraint term, a robot, a scene asset. Every recipe lists the exact files to touch. |
| 4 | [04_examples.md](04_examples.md) | A worked example for **each** part, mostly drawn from existing code (`t1-velocity`, `t1-dribbling`, `t1-maze`, `t1-soccer-maze`) plus a fully written-out new example. |
| 5 | [05_glossary_and_reference.md](05_glossary_and_reference.md) | Glossary, file-tree map, configuration registries, where the YAML / `pixi` task names point, troubleshooting traps. |

## How to read this report

- If you are **new to the repo**, read `01_components.md` then `02_interactions.md`.
- If you want to **extend something specific**, jump to the relevant recipe in
  `03_modify_each_part.md` and the matching worked example in `04_examples.md`.
- If you only want a **map of the codebase**, the tree at the top of
  `05_glossary_and_reference.md` is the fastest reference.

## Conventions used in this report

- File paths are relative to repository root (e.g. `src/colosseum/envs/colosseum_env.py`).
- A *term* is a single declarative unit (a reward term, a constraint term,
  an RMA term, …) configured by a `*Cfg` dataclass and built into a runtime
  object during environment construction.
- A *manager* is the object that owns and orchestrates terms of one type
  (`RewardManager`, `ConstraintManager`, `RmaManager`, `AbstractionManager`, …).
  Mjlab supplies the standard managers; Colosseum adds three of its own.
- The canonical environment class is `ColosseumEnv` in
  `src/colosseum/envs/colosseum_env.py`. Earlier per-feature subclasses
  (`AbstractionBasedEnv`, `ConstraintBasedEnv`, `RmaBasedEnv`, …) still exist
  as legacy aliases but new code should always use `ColosseumEnv`.

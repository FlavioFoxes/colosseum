"""Maze map definitions.

Cell encoding:
  1 / 'W' / 'w': Wall
  0:             Empty cell
  'r' / 'R':     Valid robot reset position
  'b' / 'B':     Valid ball reset position
  'g' / 'G':     Valid goal position
"""

UMAZE = [
  [1, 1, 1, 1, 1],
  [1, 0, 0, 0, 1],
  [1, "g", 0, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, "b", 0, 1],
  [1, "r", 0, 0, 1],
  [1, 1, 1, 1, 1],
]

UMAZE_TEST = [
  [1, 1, 1, 1, 1],
  [1, "g", 0, 0, 1],
  [1, 1, 1, 0, 1],
  [1, "r", 0, 0, 1],
  [1, 0, 0, 0, 1],
  [1, 1, 1, 1, 1],
]

SMALL_MAZE = [
  [1, 1, 1, 1, 1, 1, 1],
  [1, "r", 0, 1, "g", 0, 1],
  [1, 0, 0, 1, 0, 0, 1],
  [1, 1, 0, 1, 0, 1, 1],
  [1, "b", 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, "g", 1],
  [1, 1, 1, 1, 1, 1, 1],
]

MEDIUM_MAZE = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, "r", 0, 1, 1, 0, 0, 1],
  [1, 0, 0, 1, 0, 0, 0, 1],
  [1, 1, 0, "b", 0, 1, 1, 1],
  [1, 0, 0, 1, 0, 0, 0, 1],
  [1, "g", 1, 1, 0, 1, 0, 1],
  [1, 0, 0, 0, 0, 1, "g", 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
]

LARGE_MAZE = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, "r", 0, 0, 0, 1, "g", 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  [1, 0, 0, "b", 0, 0, 0, 1, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
  [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
  [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
  [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 1, "g", 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

OPEN = [
  [1, 1, 1, 1, 1],
  [1, "r", 0, 0, 1],
  [1, 0, "b", 0, 1],
  [1, 0, 0, "g", 1],
  [1, 1, 1, 1, 1],
]

# 5×5 interior open field (7×7 total with border walls, no internal walls)
OPEN_MEDIUM = [
  [1, 1, 1, 1, 1, 1, 1],
  [1, "r", 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, "b", 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, "g", 0, 1],
  [1, 1, 1, 1, 1, 1, 1],
]

# Push-only: robot starts directly behind ball, goal one cell south.
# Sokoban plan = single PUSH (no MOVE), trains the kicking mechanic in isolation.
# Robot at (2,3) is N of ball at (3,3); goal at (4,3) is S of ball.
PUSH_ONLY = [
  [1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, "r", "b", 0, "g", 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1],
]

MAPS = {
  "push_only": PUSH_ONLY,
  "umaze": UMAZE,
  "small": SMALL_MAZE,
  "medium": MEDIUM_MAZE,
  "large": LARGE_MAZE,
  "open": OPEN,
  "open_medium": OPEN_MEDIUM,
}

TEST_MAPS = {
  "umaze": UMAZE_TEST,
}

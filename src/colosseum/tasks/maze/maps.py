from typing import Tuple, List, Set
import random
import re

"""
Base algorithm:
- We start from an empty grid with borders.
- We define robot position and goal position.
- We build a road connecting the two. The road might have an arbitrary number 
    of turns (in accordance with the size of the grid) 
- We enhance this road with "elbows", that the agent can use to push the box 
    towards the goal

This generates playable levels but the problem is that they are too simplistic:
there always is a single way leading to the goal, no complex cell arrangments, 
no crossing paths...

Enhanced algorithm:
- We iterate the above procedure a given number of times. This ensures 
    crossing paths and non-trivial global solutions.
- We apply an "erosion" step: we randomly remove some cells. This increase the 
    complexity of the maxe as the agent can now perform unnecessary actions
    and therefore produce non-optimal solutions.
- We apply a "deposition" step: we randomly add some cells. This should be done
    with criterion: adding cells could block the only path leading to the solution
    (even if we iterate the procedure, it could happen that for example the terminal
    part of the paths is shared across iterations and a new cell ends up blocking 
    all of them at the same time). We can add a block only if [???]
"""

"""
Cell encoding:
  1 / 'W' / 'w': Wall
  0:             Empty cell
  'r' / 'R':     Valid robot reset position
  'g' / 'G':     Valid goal position
"""
ENCODING = {
    'WALL': '#',
    'EMPTY': '.',
    'ROBOT': 'r',
    'GOAL': '@',
    'BLOCK': 'b',
    'ELBOW': '.'
}

"""
Directions
"""
DIRECTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

"""
Fractions of the grid dimension used to define the spawn region.

These values limit random position selection to a sub-region of the
original grid. We will anchor it to the first and fourth quadrants
(this will be clear when the constants are actually used)

    row in [0, SPAWN_AREA_FRACTION_H * H)
    col in [0, SPAWN_AREA_FRACTION_W * W)

where H and W are the array height and width, SPAWN_AREA_FRACTION_H and 
SPAWN_AREA_FRACTION_W are the fractions for height and width respectively.

The resulting spawn region covers (SPAWN_AREA_FRACTION_H * SPAWN_AREA_FRACTION_W) 
of the total area (e.g. SAF_W=0.2, SAF_H=0.2 -> SPAWN_REGION=4%).
"""
ROBOT_SPAWN_AREA_FRACTION_W = 0.2
ROBOT_SPAWN_AREA_FRACTION_H = 0.2

# For the goal we will then
GOAL_SPAWN_AREA_FRACTION_W = 0.5
GOAL_SPAWN_AREA_FRACTION_H = 0.5

assert GOAL_SPAWN_AREA_FRACTION_W + ROBOT_SPAWN_AREA_FRACTION_W <= 1
assert GOAL_SPAWN_AREA_FRACTION_H + ROBOT_SPAWN_AREA_FRACTION_H <= 1

"""
Maximum and minimum grid sizes. We need to take into account that:
- we need to have at least a 1-cell border around the maze 
    (2 cells width/height reserved for this)
- we need to leave space for the path elbows
    (2 cells width/height reserved for this)
The cells introduced for the elbows are effectively playable, so the
effective playable grid size will simply be:

    MIN_GRID_WIDTH - 2
    MIN_GRID_HEIGHT - 2

"""
MIN_GRID_WIDTH, MAX_GRID_WIDTH = 6, 30
MIN_GRID_HEIGHT, MAX_GRID_HEIGHT = 6, 30


# Maze generation

def pretty_print(maze: List[List]):
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            print(maze[y][x], end='')
        print()
    print()


def maze_to_string(maze: List[List]) -> str:
    result = []

    for y in range(len(maze)):
        row = []
        for x in range(len(maze[y])):
            row.append(str(maze[y][x]))
        result.append(''.join(row))

    return '\n'.join(result) + '\n'


def is_empty(maze: List[List], row: int, col: int):
    if row < 0 or row >= len(maze) or col < 0 or col >= len(maze[0]):
        return False
    return maze[row][col] == ENCODING['EMPTY']


def is_wall(maze: List[List], row: int, col: int):
    if row < 0 or row >= len(maze) or col < 0 or col >= len(maze[0]):
        return False
    return maze[row][col] == ENCODING['WALL']


def manhattan_distance(start: Tuple[int, int], end: Tuple[int, int]):
    """Simple Manhattan distance heuristic"""
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def get_random_empty_cell(maze: List[List], avoid: Set[Tuple[int, int]] = None):
    """
    Returns a random empty cell in the specified range

    Args:
        maze: the maze, duh
        avoid: set of cells we should avoid
    """

    if not avoid:
        avoid = set()

    height, width = len(maze), len(maze[0])

    # Gather all cells
    cells = {(r, c) for r in range(0, height) for c in range(0, width)}

    # Remove non-empty cells
    cells = {(r, c) for (r, c) in cells if is_empty(maze, r, c)}

    # Remove cells in avoid set
    cells = {(r, c) for (r, c) in cells if (r, c) not in avoid}

    if len(cells) == 0:
        raise ValueError("Something weng wrong, check the maze")

    # Return a random cell among the empty ones
    return random.sample(list(cells), 1)[0]


def add_elbows(path: List[Tuple]):
    """
    Add elbows to the path:

    +--+--+--+--+--+--+--+--+--+--+
    |  |ss|  |  |  |  |  |  |  |  |
    +--+--+--+--+--+--+--+--+--+--+
    |  |pp|  |  |  |  |  |ee|ee|  |
    +--+--+--+--+--+--+--+--+--+--+
    |  |pp|pp|pp|pp|pp|pp|pp|pp|  |
    +--+--+--+--+--+--+--+--+--+--+
    |  |pp|  |  |  |  |  |  |pp|  |
    +--+--+--+--+--+--+--+--+--+--+
    |  |pp|  |  |  |  |  |  |pp|..|
    +--+--+--+--+--+--+--+--+--+--+
    |ee|pp|  |  |  |  |  |  |  |  |     e.g.    p0 = (5, 1)
    +--+--+--+--+--+--+--+--+--+--+             p1 = (6, 1)
    |ee|pp|pp|pp|pp|pp|pp|pp|pp|..|             p2 = (6, 2)
    +--+--+--+--+--+--+--+--+--+--+             d1 = (6-5, 1-1) = (1, 0) -> down
                                                d2 = (6-6, 2-1) = (0, 1) -> right

    - ss: start
    - pp: path
    - ee: elbow

    Note: elbow always involves only p0 and p1
    """

    assert len(path) >= 2, "What kind of path did you found you dumb ass"

    def has_turned(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        # Direction vectors
        d1 = (x1 - x0, y1 - y0)
        d2 = (x2 - x1, y2 - y1)

        return d1 != d2

    def get_elbow_cells(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        # d2 = (x2 - x1, y2 - y1)
        # e0 = p0 - d2
        # e1 = p1 - d2

        e0 = (x0 - x2 + x1, y0 - y2 + y1)
        e1 = (x1 - x2 + x1, y1 - y2 + y1)

        return [e0, e1]

    elbow_cells = []
    for i in range(1, len(path) - 1):
        p0 = path[i - 1]
        p1 = path[i]
        p2 = path[i + 1]
        if has_turned(p0, p1, p2):
            elbow_cells.extend(get_elbow_cells(p0, p1, p2))

    return elbow_cells


def get_box_candidate_positions(path):
    """
    Get candidate box positions. To ensure solvability, we put the box either in:
    - a corridor
    - the elbow joint
    """

    assert len(path) >= 2, "What kind of path did you found you dumb ass"

    def has_turned(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        # Direction vectors
        d1 = (x1 - x0, y1 - y0)
        d2 = (x2 - x1, y2 - y1)

        return d1 != d2

    def get_elbow_cells(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        # d2 = (x2 - x1, y2 - y1)
        # e0 = p0 - d2
        # e1 = p1 - d2

        e0 = (x0 - x2 + x1, y0 - y2 + y1)
        e1 = (x1 - x2 + x1, y1 - y2 + y1)

        return [e0, e1]

    candidate_box_positions = []
    for i in range(1, len(path) - 1):
        p0 = path[i - 1]
        p1 = path[i]
        p2 = path[i + 1]
        if has_turned(p0, p1, p2):
            candidate_box_positions.append(p1)

    return candidate_box_positions


def lerw_waypoints(
        maze: List[List],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        spread: float = 0.3,
):
    """
    Farthest-Point Sampling: each new waypoint is drawn from a distribution
    weighted by min distance^2 to all already-chosen points.
    This maximizes spatial spread across the grid.
    """

    height, width = len(maze), len(maze[0])

    # TODO check this formula
    count = min(round(spread * 6), (width * height) // 8)

    if count == 0:
        return []

    chosen = [start, goal]
    waypoints = []

    for _ in range(count):
        candidates = [
            (x, y)
            for x in range(width) for y in range(height)
            if (x, y) not in chosen
        ]
        if not candidates:
            break

        # Weight by squared min-distance from any chosen point
        weights = [min(abs(x - cx) + abs(y - cy) for cx, cy in chosen) ** 2 for x, y in candidates]
        [wp] = random.choices(candidates, weights=weights, k=1)
        waypoints.append(wp)
        chosen.append(wp)

    # Greedy nearest-neighbor ordering from start
    ordered, remaining, cur = [], list(waypoints), start
    while remaining:
        nearest = min(remaining, key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
        ordered.append(nearest)
        remaining.remove(nearest)
        cur = nearest

    return ordered


def lerw_path(
        maze: List[List],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        entropy: float = 0.3,
        spread: float = 0.3
):
    """
    Loop-Erased Random Walk with waypoints for full-grid coverage.

    Args:
        entropy: 0.0 = greedy straight line, 1.0 = fully random local walk
        spread: 0.0 = direct (original behavior), 1.0 = many waypoints, full coverage
    """

    height, width = len(maze), len(maze[0])

    def neighbors(x, y):
        return [(x + dx, y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= x + dx < width and 0 <= y + dy < height]

    def biased_step(cell, target):
        opts = neighbors(*cell)
        if random.random() > entropy:
            return min(opts, key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]))
        return random.choice(opts)

    def current_target_idx(targets, visited):
        return next((i for i, t in enumerate(targets) if t not in visited), len(targets))

    waypoints = lerw_waypoints(maze, start, goal, spread)
    targets = waypoints + [goal]

    path = [start]
    visited = {start: 0}

    while True:
        ti = current_target_idx(targets, visited)
        if ti >= len(targets):
            break

        current = path[-1]
        nxt = biased_step(current, targets[ti])

        if nxt in visited:
            loop_index = visited[nxt]
            for cell in path[loop_index + 1:]:
                del visited[cell]
            path = path[:loop_index + 1]
        else:
            visited[nxt] = len(path)
            path.append(nxt)

    return path


def apply_erosion(maze: List[List], strength: float):
    """Randomly remove walls to increase complexity and open up the maze"""

    height, width = len(maze), len(maze[0])
    playable_cells = sum([
        1 if maze[r][c] == ENCODING['EMPTY'] or maze[r][c] == ENCODING['ELBOW'] else 0
        for c in range(width) for r in range(height)
    ]
    )
    num_cells_to_erode = int(playable_cells * strength)

    while num_cells_to_erode > 0:

        # Pick a random cell
        r = random.randint(0, height - 1)
        c = random.randint(0, width - 1)

        if maze[r][c] == ENCODING['EMPTY']:
            continue

        if maze[r][c] == ENCODING['ROBOT']:
            continue

        if maze[r][c] == ENCODING['GOAL']:
            continue

        # Check all possible neighbors
        neighbors = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

        # Keep only the existing ones
        neighbors = set([neighbor for neighbor in neighbors if neighbor[0] < height and neighbor[1] < width])

        num_empty_neighbors = sum(
            [1 if maze[neighbor[0]][neighbor[1]] == ENCODING['EMPTY'] else 0 for neighbor in neighbors])

        # Accept the cell only if a neighbor is empty
        # We can even be more restrictive and keep only cells with at least 2 empty neighbors
        if num_empty_neighbors < 1:
            continue

        # Finally, erosion could happen
        maze[r][c] = ENCODING['EMPTY']

        # If erosion successful, decrement counter
        num_cells_to_erode -= 1


def apply_deposition(maze: List[List], strength: float):
    raise ValueError("Not implemented")


def add_borders(maze: List[List]):
    """Add wall borders around the maze"""

    height, width = len(maze), len(maze[0])
    new_height, new_width = height + 2, width + 2

    new_maze = [[ENCODING['EMPTY'] for _ in range(new_width)] for _ in range(new_height)]

    # Top and bottom borders
    for c in range(new_width):
        new_maze[0][c] = ENCODING['WALL']
        new_maze[new_height - 1][c] = ENCODING['WALL']

    # Left and right borders
    for r in range(new_height):
        new_maze[r][0] = ENCODING['WALL']
        new_maze[r][new_width - 1] = ENCODING['WALL']

    # Copy old maze into new one
    for c in range(width):
        for r in range(height):
            new_maze[c + 1][r + 1] = maze[c][r]

    return new_maze


def generate_maze(
        height: int,
        width: int,
        erosion: bool = False,
        deposition: bool = False,
        num_road_iterations: int = 1,
        erosion_strength: float = 0.5,
        deposition_strength: float = 0.05,
        seed: int = None,
        **kwargs
) -> List[List]:
    """
    Generate a random maze using recursive backtracking.
    The maze will feature a single block and a single goal position.
    """

    random.seed(seed)

    assert MIN_GRID_WIDTH <= width <= MAX_GRID_WIDTH, f"Maze width out of bounds({width}), should be {MIN_GRID_WIDTH} <= x <= {MAX_GRID_WIDTH}."
    assert MIN_GRID_HEIGHT <= height <= MAX_GRID_HEIGHT, f"Maze height out of bounds ({height}), should be {MIN_GRID_HEIGHT} <= y <= {MAX_GRID_HEIGHT}."

    full_width, full_height = width, height
    workable_width, workable_height = width - 4, height - 4

    # Generate a grid full of wall cells
    full_maze = [[ENCODING['WALL'] for _ in range(full_width)] for _ in range(full_height)]
    workable_maze = [[ENCODING['WALL'] for _ in range(workable_width)] for _ in range(workable_height)]

    # Define robot position
    robot_x = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_W * workable_width)
    robot_y = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_H * workable_height)
    robot_position = (robot_x, robot_y)
    workable_maze[robot_x][robot_y] = ENCODING['ROBOT']

    # Define goal position
    goal_x = workable_width - int(random.random() * GOAL_SPAWN_AREA_FRACTION_W * workable_width) - 1
    goal_y = workable_height - int(random.random() * GOAL_SPAWN_AREA_FRACTION_H * workable_height) - 1
    goal_position = (goal_x, goal_y)
    workable_maze[goal_x][goal_y] = ENCODING['GOAL']

    # Build roads connecting start to goal
    paths = []
    for _ in range(num_road_iterations):
        path = lerw_path(workable_maze, robot_position, goal_position, **kwargs)
        paths.append(path)

    # Get elbow cells
    elbows = []
    for path in paths:
        elbows.append(add_elbows(path))

    # Get positions where we can put the box
    boxes = []
    for path in paths:
        boxes.append(get_box_candidate_positions(path))

    # Dump all the stuff on the new maze
    full_maze[robot_x + 2][robot_y + 2] = ENCODING['ROBOT']
    full_maze[goal_x + 2][goal_y + 2] = ENCODING['GOAL']

    path_cells = set([cell for path in paths for cell in path])
    elbow_cells = set([cell for elbow in elbows for cell in elbow])
    box_cells = set([cell for box in boxes for cell in box]) - {robot_position} - {goal_position}
    cells = path_cells.union(elbow_cells)

    for cell in cells:
        if cell != goal_position and cell != robot_position:
            row, col = cell
            if cell in path_cells:
                full_maze[row + 2][col + 2] = ENCODING['EMPTY']
            else:
                full_maze[row + 2][col + 2] = ENCODING['ELBOW']

    # Apply erosion. This does not invalidate any solution to the problem,
    # it just adds maze complexity
    if erosion:
        apply_erosion(full_maze, erosion_strength)

    # Apply deposition
    if deposition:
        apply_deposition(full_maze, deposition_strength)

    # Any free cell at this point is a candidate for the block,
    # but we need to carefully analyze where to put it as this can
    # invalidate the solution.
    # Solution: place the block on p1 or in corridors
    #       (we know for sure we can push the box from there)
    # block_x, block_y = get_random_empty_cell(maze, avoid=elbow_cells)
    block_x, block_y = random.sample(sorted(box_cells), 1)[0]
    block_position = (block_x, block_y)
    if block_position == (-1, -1):
        raise ValueError(f"Cannot place block, check the maze.")
    full_maze[block_x + 2][block_y + 2] = ENCODING['BLOCK']

    # Enforce borders (erosion could breach the maze)
    for r in range(full_height):
        full_maze[r][0] = ENCODING['WALL']
        full_maze[r][full_width - 1] = ENCODING['WALL']

    for c in range(full_width):
        full_maze[0][c] = ENCODING['WALL']
        full_maze[full_height - 1][c] = ENCODING['WALL']

    return full_maze


def generate_base_maze(
        height: int,
        width: int,
        erosion: bool = False,
        deposition: bool = False,
        num_road_iterations: int = 1,
        erosion_strength: float = 0.5,
        deposition_strength: float = 0.05,
        seed: int = None,
        **kwargs
    ):
    """
    Similar to generate_maze, but does NOT place a box.
    Returns (base_maze, box_cells) so callers can inject any valid box position.

    box_cells is the set of elbow-joint cells (turn points on the path),
    which guarantees solvability when a box is placed there.
    Cells that coincide with robot_position or goal_position are excluded.
    """

    random.seed(seed)

    assert MIN_GRID_WIDTH <= width <= MAX_GRID_WIDTH, f"Maze width out of bounds({width}), should be {MIN_GRID_WIDTH} <= x <= {MAX_GRID_WIDTH}."
    assert MIN_GRID_HEIGHT <= height <= MAX_GRID_HEIGHT, f"Maze height out of bounds ({height}), should be {MIN_GRID_HEIGHT} <= y <= {MAX_GRID_HEIGHT}."

    full_width, full_height = width, height
    workable_width, workable_height = width - 4, height - 4

    # Generate a grid full of wall cells
    full_maze = [[ENCODING['WALL'] for _ in range(full_width)] for _ in range(full_height)]
    workable_maze = [[ENCODING['WALL'] for _ in range(workable_width)] for _ in range(workable_height)]

    # Define robot position
    robot_x = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_W * workable_width)
    robot_y = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_H * workable_height)
    robot_position = (robot_x, robot_y)
    workable_maze[robot_x][robot_y] = ENCODING['ROBOT']

    # Define goal position
    goal_x = workable_width - int(random.random() * GOAL_SPAWN_AREA_FRACTION_W * workable_width) - 1
    goal_y = workable_height - int(random.random() * GOAL_SPAWN_AREA_FRACTION_H * workable_height) - 1
    goal_position = (goal_x, goal_y)
    workable_maze[goal_x][goal_y] = ENCODING['GOAL']

    # Build roads connecting start to goal
    paths = [
        lerw_path(workable_maze, robot_position, goal_position, **kwargs)
        for _ in range(num_road_iterations)
    ]

    # Get elbow cells and box candidate cells
    elbows = [add_elbows(p) for p in paths]
    boxes = [get_box_candidate_positions(p) for p in paths]

    path_cells = {cell for path  in paths  for cell in path}
    elbow_cells = {cell for elbow in elbows for cell in elbow}
    box_cells = (
        {cell for box in boxes for cell in box}
        - {robot_position}
        - {goal_position}
    )

    # Dump all the stuff on the new maze
    full_maze[robot_x + 2][robot_y + 2] = ENCODING['ROBOT']
    full_maze[goal_x + 2][goal_y + 2] = ENCODING['GOAL']

    for cell in path_cells | elbow_cells:
        if cell in (robot_position, goal_position):
            continue
        row, col = cell
        full_maze[row + 2][col + 2] = (
            ENCODING['EMPTY'] if cell in path_cells else ENCODING['ELBOW']
        )

    # Apply erosion. This does not invalidate any solution to the problem,
    # it just adds maze complexity
    if erosion:
        apply_erosion(full_maze, erosion_strength)

    # Apply deposition
    if deposition:
        apply_deposition(full_maze, deposition_strength)

    # Enforce borders (erosion could breach the maze)
    for r in range(full_height):
        full_maze[r][0] = ENCODING['WALL']
        full_maze[r][full_width -1] = ENCODING['WALL']
    for c in range(full_width):
        full_maze[0][c] = ENCODING['WALL']
        full_maze[full_height - 1][c] = ENCODING['WALL']

    # Translate box_cells to full-maze coordinates (+2 offset)
    box_cells_full = {(r + 2, c + 2) for (r, c) in box_cells}

    return full_maze, box_cells_full


def generate_empty_maze(
    height: int,
    width: int,
    erosion: bool = False,
    deposition: bool = False,
    num_road_iterations: int = 1,
    erosion_strength: float = 0.5,
    deposition_strength: float = 0.05,
    seed: int = None,
    **kwargs
):
    """
    Similar to generate_maze, but doesn't place robot, goal or box.
    Returns base_maze so callers can inject any robot, goal or box position.
    """

    random.seed(seed)

    assert MIN_GRID_WIDTH <= width <= MAX_GRID_WIDTH, f"Maze width out of bounds({width}), should be {MIN_GRID_WIDTH} <= x <= {MAX_GRID_WIDTH}."
    assert MIN_GRID_HEIGHT <= height <= MAX_GRID_HEIGHT, f"Maze height out of bounds ({height}), should be {MIN_GRID_HEIGHT} <= y <= {MAX_GRID_HEIGHT}."

    full_width, full_height = width, height
    workable_width, workable_height = width - 4, height - 4

    # Generate a grid full of wall cells
    full_maze = [[ENCODING['WALL'] for _ in range(full_width)] for _ in range(full_height)]
    workable_maze = [[ENCODING['WALL'] for _ in range(workable_width)] for _ in range(workable_height)]

    # Define robot position
    robot_x = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_W * workable_width)
    robot_y = int(random.random() * ROBOT_SPAWN_AREA_FRACTION_H * workable_height)
    robot_position = (robot_x, robot_y)

    # Define goal position
    goal_x = workable_width - int(random.random() * GOAL_SPAWN_AREA_FRACTION_W * workable_width) - 1
    goal_y = workable_height - int(random.random() * GOAL_SPAWN_AREA_FRACTION_H * workable_height) - 1
    goal_position = (goal_x, goal_y)

    # Build roads connecting start to goal
    paths = [
        lerw_path(workable_maze, robot_position, goal_position, **kwargs)
        for _ in range(num_road_iterations)
    ]

    # Get elbow cells
    elbows = [add_elbows(p) for p in paths]

    path_cells = {cell for path  in paths  for cell in path}
    elbow_cells = {cell for elbow in elbows for cell in elbow}

    # Dump all the stuff on the new maze
    for cell in path_cells | elbow_cells:
        if cell in (robot_position, goal_position):
            continue
        row, col = cell
        full_maze[row + 2][col + 2] = (
            ENCODING['EMPTY'] if cell in path_cells else ENCODING['ELBOW']
        )

    # Apply erosion. This does not invalidate any solution to the problem,
    # it just adds maze complexity
    if erosion:
        apply_erosion(full_maze, erosion_strength)

    # Apply deposition
    if deposition:
        apply_deposition(full_maze, deposition_strength)

    # Enforce borders (erosion could breach the maze)
    for r in range(full_height):
        full_maze[r][0] = ENCODING['WALL']
        full_maze[r][full_width -1] = ENCODING['WALL']
    for c in range(full_width):
        full_maze[0][c] = ENCODING['WALL']
        full_maze[full_height - 1][c] = ENCODING['WALL']

    return full_maze


# Simulate plan

def parse_plan_action(action_string: str) -> Tuple[str, List[str]]:
    # Extract action name and parameters
    # Example: "push-box(loc-4-5, loc-13-7, loc-13-6, loc-13-5)"
    match = re.match(r'([\w-]+)\((.*)\)', action_string.strip())
    if not match:
        return None, None

    action_name = match.group(1)
    params_str = match.group(2)
    params = [p.strip() for p in params_str.split(',')]

    return action_name, params


def location_to_coords(loc_id) -> Tuple[int, int]:
    # Convert "loc-x-y" to (x, y)
    parts = str(loc_id).split('-')
    x, y = int(parts[1]), int(parts[2])
    return x, y


def simulate_plan(initial_board: List[List[str]], plan_actions: List) -> List[List[List[str]]]:
    # Track goal positions separately so they persist when blocks move
    height, width = len(initial_board), len(initial_board[0])
    goal_positions = set()

    for y in range(height):
        for x in range(width):
            if initial_board[y][x] == ENCODING['GOAL']:
                goal_positions.add((x, y))

    # Initialize with the starting board state
    board_states = []
    current_board = [row[:] for row in initial_board]  # Deep copy
    board_states.append([row[:] for row in current_board])

    # Process each action
    for action in plan_actions:

        # action_name = action.action.name
        # params = action.action.parameters
        action_name, params = parse_plan_action(str(action))
        print(f"Action name: {action_name}")
        print(f"Action params: {params}")
        print()

        if action_name == "push-box":
            # Handle both 3-param (no axioms) and 4-param (with axioms) versions
            if len(params) == 3:
                # Without axioms: push-box(x, y, z)
                # x = robot position, y = box position, z = new box position
                x, y, z = params
                x_coords = location_to_coords(x)
                y_coords = location_to_coords(y)
                z_coords = location_to_coords(z)

                # Clear old player position (restore goal if it was there)
                if x_coords in goal_positions:
                    current_board[x_coords[1]][x_coords[0]] = ENCODING['GOAL']
                else:
                    current_board[x_coords[1]][x_coords[0]] = ENCODING['EMPTY']

                # Clear old box position (restore goal if it was there)
                if y_coords in goal_positions:
                    current_board[y_coords[1]][y_coords[0]] = ENCODING['GOAL']
                else:
                    current_board[y_coords[1]][y_coords[0]] = ENCODING['EMPTY']

                # Move player to y position
                current_board[y_coords[1]][y_coords[0]] = ENCODING['ROBOT']

                # Move box to z position
                current_board[z_coords[1]][z_coords[0]] = ENCODING['BLOCK']

            elif len(params) == 4:
                # With axioms: push-box(l, x, y, z)
                # l = reachable position, x = position adjacent to box, y = box position, z = new box position
                l, x, y, z = params
                x_coords = location_to_coords(x)
                y_coords = location_to_coords(y)
                z_coords = location_to_coords(z)

                # Clear old player position (restore goal if it was there)
                # Note: player could be anywhere reachable, we find it on the board
                # For now, assume player is at x (adjacent to box)
                if x_coords in goal_positions:
                    current_board[x_coords[1]][x_coords[0]] = ENCODING['GOAL']
                else:
                    current_board[x_coords[1]][x_coords[0]] = ENCODING['EMPTY']

                # Clear old box position (restore goal if it was there)
                if y_coords in goal_positions:
                    current_board[y_coords[1]][y_coords[0]] = ENCODING['GOAL']
                else:
                    current_board[y_coords[1]][y_coords[0]] = ENCODING['EMPTY']

                # Move player to y position
                current_board[y_coords[1]][y_coords[0]] = ENCODING['ROBOT']

                # Move box to z position
                current_board[z_coords[1]][z_coords[0]] = ENCODING['BLOCK']

        elif action_name == "move":
            # Parameters: fr (from location), to (to location)
            if len(params) == 2:
                fr, to = params
                fr_coords = location_to_coords(fr)
                to_coords = location_to_coords(to)

                # Clear old position (restore goal if it was there)
                if fr_coords in goal_positions:
                    current_board[fr_coords[1]][fr_coords[0]] = ENCODING['GOAL']
                else:
                    current_board[fr_coords[1]][fr_coords[0]] = ENCODING['EMPTY']

                # Move player to new position
                current_board[to_coords[1]][to_coords[0]] = ENCODING['ROBOT']

        # Add current board state to history
        board_states.append([row[:] for row in current_board])

    return board_states


"""
Collection of predefined maps.
WARN: these should be defined according to the ENCODING.
"""

UMAZE = [
  ['#', '#', '#', '#', '#'],
  ['#', '@', '.', '@', '#'],
  ['#', '#', '#', '@', '#'],
  ['#', 'r', '.', '.', '#'],
  ['#', '#', '#', '#', '#'],
]

UMAZE_TEST = [
  ['#', '#', '#', '#', '#'],
  ['#', '.', '@', '.', '#'],
  ['#', '#', '#', '.', '#'],
  ['#', 'r', '.', '@', '#'],
  ['#', '#', '#', '#', '#'],
]

SMALL_MAZE = [
  ['#', '#', '#', '#', '#', '#', '#'],
  ['#', 'r', '.', '#', '@', '.', '#'],
  ['#', '.', '.', '#', '.', '.', '#'],
  ['#', '#', '.', '#', '.', '#', '#'],
  ['#', '.', '.', '.', '.', '.', '#'],
  ['#', '.', '#', '#', '#', '@', '#'],
  ['#', '#', '#', '#', '#', '#', '#'],
]

MEDIUM_MAZE = [
  ['#', '#', '#', '#', '#', '#', '#', '#'],
  ['#', 'r', '.', '#', '#', '.', '.', '#'],
  ['#', '.', '.', '#', '.', '.', '.', '#'],
  ['#', '#', '.', '.', '.', '#', '#', '#'],
  ['#', '.', '.', '#', '.', '.', '.', '#'],
  ['#', '@', '#', '#', '.', '#', '.', '#'],
  ['#', '.', '.', '.', '.', '#', '@', '#'],
  ['#', '#', '#', '#', '#', '#', '#', '#'],
]

LARGE_MAZE = [
  ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
  ['#', 'r', '.', '.', '.', '#', '@', '.', '.', '.', '.', '#'],
  ['#', '.', '#', '#', '.', '#', '.', '#', '.', '#', '.', '#'],
  ['#', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#'],
  ['#', '.', '#', '#', '#', '#', '.', '#', '#', '#', '.', '#'],
  ['#', '.', '.', '#', '.', '.', '.', '.', '#', '.', '.', '#'],
  ['#', '#', '.', '#', '.', '#', '#', '.', '.', '.', '#', '#'],
  ['#', '.', '.', '#', '.', '.', '.', '.', '#', '.', '.', '#'],
  ['#', '.', '#', '.', '.', '#', '#', '#', '#', '#', '.', '#'],
  ['#', '.', '#', '#', '#', '.', '.', '.', '.', '#', '.', '#'],
  ['#', '.', '.', '.', '.', '.', '#', '@', '.', '.', '.', '#'],
  ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
]

OPEN = [
  ['#', '#', '#', '#', '#'],
  ['#', 'r', '.', '.', '#'],
  ['#', '.', '.', '.', '#'],
  ['#', '.', '.', '@', '#'],
  ['#', '#', '#', '#', '#'],
]

PREDEFINED_MAPS = {
  "umaze": UMAZE,
  "small": SMALL_MAZE,
  "medium": MEDIUM_MAZE,
  "large": LARGE_MAZE,
  "open": OPEN,
}

TEST_MAPS = {
  "umaze": UMAZE_TEST,
}

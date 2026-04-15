from unified_planning.shortcuts import *
from itertools import permutations
from typing import Tuple
import copy

from colosseum.tasks.maze.maps import ENCODING
from colosseum.tasks.maze.maps import (
    generate_base_maze,
    generate_empty_maze,
    maze_to_string,
    simulate_plan,
    is_wall
)

"""
Solver-related stuff.Usage example:

# Generate a map
sokoban_level = generate_map(width, height, erosion=erosion)

# Transform the sokoban level to string, format expected from the solver
stringified_level = maze_to_string(sokoban_level)

# Solve the problem
sokoban_problem = create_sokoban_problem(stringified_level, with_axioms=True)
solution = solve_sokoban_problem(sokoban_problem)
"""


def create_sokoban_problem(level: str, with_axioms: bool = False):

    problem = up.model.Problem("sokoban")
    loc = UserType("location")

    has_player, has_box, adjacent, adjacent_2, can_reach = (
        up.model.Fluent("has_player", BoolType(), l=loc),
        up.model.Fluent("has_box", BoolType(), l=loc),
        up.model.Fluent("adjacent", BoolType(), l1=loc, l2=loc),
        up.model.Fluent("adjacent_2", BoolType(), l1=loc, l2=loc),
        up.model.Fluent("can_reach", DerivedBoolType(), l=loc) if with_axioms else None,
    )

    for fluent in [has_player, has_box, adjacent, adjacent_2, can_reach]:
        if fluent:
            problem.add_fluent(fluent, default_initial_value=False)

    if with_axioms:
        # Axiom 1
        a1 = up.model.Axiom("reach-axiom-1", l=loc)
        a1.set_head(can_reach(a1.parameters[0]))
        a1.add_body_condition(has_player(a1.parameters[0]))
        problem.add_axiom(a1)

        # Axiom 2
        a2 = up.model.Axiom("reach-axiom-2", to=loc)
        to, fr = a2.parameters[0], Variable("from", loc)
        a2.set_head(can_reach(to))
        a2.add_body_condition(Not(has_box(to)))
        a2.add_body_condition(
            Exists(
                And(
                    can_reach(fr),
                    adjacent(fr, to)
                ),
                fr
            )
        )
        problem.add_axiom(a2)

    cost_dict = {}

    if not with_axioms:
        # Move action
        move = up.model.InstantaneousAction("move", fr=loc, to=loc)
        fr, to = move.parameters
        move.add_precondition(adjacent(fr, to))
        move.add_precondition(has_player(fr))
        move.add_precondition(Not(has_box(to)))
        move.add_effect(has_player(fr), False)
        move.add_effect(has_player(to), True)
        cost_dict[move] = 0
        problem.add_action(move)

        # Push box
        push_box = up.model.InstantaneousAction("push-box", x=loc, y=loc, z=loc)
        x, y, z = push_box.parameters
        push_box.add_precondition(adjacent(x, y))
        push_box.add_precondition(adjacent(y, z))
        push_box.add_precondition(adjacent_2(x, z))
        push_box.add_precondition(has_player(x))
        push_box.add_precondition(has_box(y))
        push_box.add_precondition(Not(has_box(z)))
        push_box.add_effect(has_player(x), False)
        push_box.add_effect(has_player(y), True)
        push_box.add_effect(has_box(y), False)
        push_box.add_effect(has_box(z), True)
        cost_dict[push_box] = 1
        problem.add_action(push_box)

    if with_axioms:
        # Push box
        push_box = up.model.InstantaneousAction("push-box", l=loc, x=loc, y=loc, z=loc)
        l, x, y, z = push_box.parameters
        push_box.add_precondition(adjacent(x, y))
        push_box.add_precondition(adjacent(y, z))
        push_box.add_precondition(adjacent_2(x, z))
        push_box.add_precondition(has_player(l))
        push_box.add_precondition(can_reach(x))
        push_box.add_precondition(has_box(y))
        push_box.add_precondition(Not(has_box(z)))
        push_box.add_effect(has_player(l), False)
        push_box.add_effect(has_player(y), True)
        push_box.add_effect(has_box(y), False)
        push_box.add_effect(has_box(z), True)
        cost_dict[push_box] = 1
        problem.add_action(push_box)

    problem.add_quality_metric(up.model.metrics.MinimizeActionCosts(cost_dict))

    # Encode level
    level_array = [[char for char in line] for line in level.splitlines()]
    num_col, num_row = len(level_array[0]), len(level_array)

    loc_objs = {}

    for x in range(num_col):
        for y in range(num_row):
            entry = level_array[y][x]
            if entry != ENCODING['WALL']:
                loc_id = f"loc-{x}-{y}"
                cur_obj = up.model.Object(loc_id, loc)
                loc_objs[loc_id] = cur_obj
                problem.add_object(cur_obj)

                if entry == ENCODING['ROBOT']:
                    problem.set_initial_value(has_player(cur_obj), True)
                elif entry == ENCODING['BLOCK']:
                    problem.set_initial_value(has_box(cur_obj), True)
                elif entry == ENCODING['GOAL']:
                    problem.add_goal(has_box(cur_obj))

    # Set adjacency relationships
    for obj1 in loc_objs:
        for obj2 in loc_objs:
            x1, y1 = map(int, str(obj1).split("-")[1:])
            x2, y2 = map(int, str(obj2).split("-")[1:])
            distance = abs(x1 - x2) + abs(y1 - y2)

            if distance == 1:
                problem.set_initial_value(adjacent(loc_objs[obj1], loc_objs[obj2]), True)
            elif distance == 2 and (x1 == x2 or y1 == y2):
                problem.set_initial_value(adjacent_2(loc_objs[obj1], loc_objs[obj2]), True)

    return problem


def solve_sokoban_problem(problem):
    """
    Solve the sokoban problem associated to the level
    """

    # Solve the problem
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)  # , output_stream=sys.stdout
        if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
            print(f"{planner.name} found this plan:")
            return result.plan
        else:
            return None


def place_box(base_maze: List[List], row: int, col: int) -> List[List]:
    """Return a deep copy of base_maze with a box placed at (row, col)."""
    maze = [r[:] for r in base_maze]
    maze[row][col] = ENCODING['BLOCK']
    return maze


def generate_transition_pairs(
        height: int,
        width: int,
        with_axioms: bool = True,
        go_stupid: bool = False,
        **kwargs
    ) -> List[Tuple[List[List], List[List]]]:
    """
    Generate (initial_state, first_step_state) pairs.

    The robot and goal positions are fixed; only the box position varies.
    Each variant is solved and only solvable ones (plan length >= 1) are kept.

    if go_stupid: ignore all heuristics and return fully random mazes.
    """

    mazes = []
    if not go_stupid:
        base_maze, box_cells = generate_base_maze(
            height=height,
            width=width,
            **kwargs
        )

        if not box_cells:
            raise ValueError("No valid box positions found — try a larger grid or more road iterations.")

        for (row, col) in box_cells:
            maze = place_box(base_maze, row, col)
            mazes.append(maze)

    else:

        empty_maze = generate_empty_maze(
            height=height,
            width=width,
            **kwargs
        )

        # Gather empty cells
        cells = {(row, col) for row in range(height) for col in range(width)}
        empty_cells = {cell for cell in cells if not is_wall(empty_maze, *cell)}

        # Generate ALL combinations of (robot, box, goal) across empty cells
        for robot_position, box_position, goal_position in permutations(sorted(empty_cells), 3):
            maze_copy = copy.deepcopy(empty_maze)
            maze_copy[robot_position[0]][robot_position[1]] = ENCODING['ROBOT']
            maze_copy[box_position[0]][box_position[1]] = ENCODING['BLOCK']
            maze_copy[goal_position[0]][goal_position[1]] = ENCODING['GOAL']
            mazes.append(maze_copy)

    # At this point we have a set of mazes

    pairs = []
    for maze in mazes:
        problem = create_sokoban_problem(maze_to_string(maze), with_axioms=with_axioms)
        plan = solve_sokoban_problem(problem)

        # No feasible plan
        if plan:

            actions = plan.actions

            board_states = simulate_plan(maze, actions[:1])
            pairs.append((board_states[0], board_states[1]))

        else:
            # Add no-op
            pairs.append((maze, maze))

    return pairs

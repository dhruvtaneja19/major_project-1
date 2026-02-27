"""
Space-Time A* Search for single-agent path planning on a grid.

Each state is (x, y, t) — a cell at a specific time step.
The search respects:
  - Vertex constraints: (x, y, t) is blocked
  - Edge constraints: moving from (x1,y1,t) to (x2,y2,t+1) is blocked
  - Energy-aware cost function: penalizes distance, turns, waits, congestion

This module is used by the CBS (Conflict-Based Search) high-level solver.
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

# Directions: (dx, dy) — 4-connected grid + wait action
MOVES = [
    (0, 0),    # wait
    (1, 0),    # east
    (-1, 0),   # west
    (0, 1),    # north
    (0, -1),   # south
]

# State = (x, y, t)
State = Tuple[int, int, int]
Position = Tuple[int, int]


class VertexConstraint:
    """Robot cannot be at (x, y) at time t."""
    __slots__ = ('x', 'y', 't')

    def __init__(self, x: int, y: int, t: int):
        self.x = x
        self.y = y
        self.t = t

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t

    def __hash__(self):
        return hash((self.x, self.y, self.t))

    def __repr__(self):
        return f'V({self.x},{self.y},t={self.t})'


class EdgeConstraint:
    """Robot cannot move from (x1,y1) to (x2,y2) between time t and t+1."""
    __slots__ = ('x1', 'y1', 'x2', 'y2', 't')

    def __init__(self, x1: int, y1: int, x2: int, y2: int, t: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.t = t

    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1 and
                self.x2 == other.x2 and self.y2 == other.y2 and
                self.t == other.t)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.t))

    def __repr__(self):
        return f'E(({self.x1},{self.y1})->({self.x2},{self.y2}),t={self.t})'


class EnergyAwareCost:
    """
    Energy-aware cost model for space-time A*.

    Cost = w_dist * distance
         + w_turn * num_turns
         + w_wait * idle_time
         + w_cong * congestion_penalty

    Congestion is computed from the reservation table of other agents.
    """

    def __init__(self, w_dist: float = 1.0, w_turn: float = 0.3,
                 w_wait: float = 0.2, w_cong: float = 0.4):
        self.w_dist = w_dist
        self.w_turn = w_turn
        self.w_wait = w_wait
        self.w_cong = w_cong

    def step_cost(self, prev_dir: Optional[Tuple[int, int]],
                  dx: int, dy: int,
                  congestion_map: Optional[Dict[Position, int]] = None,
                  nx: int = 0, ny: int = 0) -> float:
        """Compute the cost of a single step."""
        cost = 0.0

        # Distance cost
        if dx == 0 and dy == 0:
            # Wait action
            cost += self.w_wait * 1.0
        else:
            cost += self.w_dist * 1.0  # unit step on grid

        # Turn penalty
        if prev_dir is not None and (dx != 0 or dy != 0):
            if prev_dir != (dx, dy):
                cost += self.w_turn * 1.0

        # Congestion penalty
        if congestion_map and (nx, ny) in congestion_map:
            count = congestion_map[(nx, ny)]
            cost += self.w_cong * count

        return cost


class SpaceTimeAStar:
    """
    Space-Time A* search on a 2D grid.

    The grid is represented as a set of obstacle positions.
    Constraints are imposed by the CBS high-level solver.
    """

    def __init__(self, grid_width: int, grid_height: int,
                 obstacles: Set[Position],
                 cost_model: Optional[EnergyAwareCost] = None,
                 max_time: int = 100):
        self.width = grid_width
        self.height = grid_height
        self.obstacles = obstacles
        self.cost = cost_model or EnergyAwareCost()
        self.max_time = max_time

    def search(self, start: Position, goal: Position,
               vertex_constraints: Set[VertexConstraint] = None,
               edge_constraints: Set[EdgeConstraint] = None,
               congestion_map: Optional[Dict[Position, int]] = None
               ) -> Optional[List[Position]]:
        """
        Find an optimal conflict-free path from start to goal in space-time.

        Returns: list of (x, y) positions at each timestep, or None if no
                 path exists within max_time.
        """
        if vertex_constraints is None:
            vertex_constraints = set()
        if edge_constraints is None:
            edge_constraints = set()
        if congestion_map is None:
            congestion_map = {}

        # Build a set for fast constraint lookup
        v_set = {(c.x, c.y, c.t) for c in vertex_constraints}
        e_set = {(c.x1, c.y1, c.x2, c.y2, c.t) for c in edge_constraints}

        # Heuristic: Manhattan distance (admissible for 4-connected grid)
        def h(x, y):
            return abs(x - goal[0]) + abs(y - goal[1])

        # Open list: (f_cost, g_cost, x, y, t, prev_direction)
        start_state = (h(start[0], start[1]), 0.0, start[0], start[1], 0, None)
        open_list = [start_state]
        # Closed set: (x, y, t)
        closed = set()
        # Parent map for path reconstruction
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        counter = 0  # tie-breaking

        while open_list:
            f, g, x, y, t, prev_dir = heapq.heappop(open_list)

            if (x, y) == goal and t > 0:
                # Reconstruct path
                path = self._reconstruct(came_from, (x, y, t), start)
                return path

            if (x, y, t) in closed:
                continue
            closed.add((x, y, t))

            if t >= self.max_time:
                continue

            for dx, dy in MOVES:
                nx, ny, nt = x + dx, y + dy, t + 1

                # Bounds check
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                # Obstacle check
                if (nx, ny) in self.obstacles:
                    continue

                # Vertex constraint check
                if (nx, ny, nt) in v_set:
                    continue

                # Edge constraint check (swap conflict)
                if (x, y, nx, ny, t) in e_set:
                    continue

                if (nx, ny, nt) in closed:
                    continue

                # Compute cost
                step_g = self.cost.step_cost(prev_dir, dx, dy, congestion_map,
                                             nx, ny)
                new_g = g + step_g
                new_f = new_g + h(nx, ny)

                new_dir = (dx, dy) if (dx != 0 or dy != 0) else prev_dir

                heapq.heappush(open_list,
                               (new_f, new_g, nx, ny, nt, new_dir))
                if (nx, ny, nt) not in came_from:
                    came_from[(nx, ny, nt)] = (x, y, t)

        return None  # No path found

    def _reconstruct(self, came_from, goal_state, start):
        """Reconstruct path from came_from map."""
        path = []
        current = goal_state
        while current is not None:
            path.append((current[0], current[1]))
            if (current[0], current[1]) == start and current[2] == 0:
                break
            current = came_from.get(current)
        path.reverse()
        return path

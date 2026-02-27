"""
Conflict-Based Search (CBS) — High-Level Multi-Agent Path Planner.

CBS is a two-level algorithm:
  High level: Binary constraint tree — detects conflicts between agents
              and resolves them by adding vertex/edge constraints.
  Low level:  Space-Time A* — replans individual agents under constraints.

This implementation includes:
  - Vertex conflict detection: two agents at same (x,y) at same time
  - Edge conflict detection: two agents swapping positions
  - Energy-aware cost function integration
  - Congestion-aware replanning
  - Configurable for 3-5 agents

Reference: Sharon et al., "Conflict-Based Search for Optimal
           Multi-Agent Pathfinding," Artificial Intelligence, 2015.
"""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .space_time_astar import (
    EdgeConstraint,
    EnergyAwareCost,
    Position,
    SpaceTimeAStar,
    VertexConstraint,
)


@dataclass
class Conflict:
    """A conflict between two agents."""
    agent_i: int
    agent_j: int
    conflict_type: str  # 'vertex' or 'edge'
    x1: int
    y1: int
    x2: int = 0
    y2: int = 0
    timestep: int = 0


@dataclass(order=True)
class CTNode:
    """A node in the Constraint Tree."""
    cost: float
    id: int = field(compare=True)
    constraints: Dict[int, Tuple[Set[VertexConstraint], Set[EdgeConstraint]]] = \
        field(default_factory=dict, compare=False, repr=False)
    solution: Dict[int, List[Position]] = \
        field(default_factory=dict, compare=False, repr=False)
    conflicts: List[Conflict] = \
        field(default_factory=list, compare=False, repr=False)


class CBS:
    """
    Conflict-Based Search for Multi-Agent Path Finding.

    Finds conflict-free paths for all agents on a shared grid,
    minimizing total energy-aware cost.
    """

    def __init__(self, grid_width: int, grid_height: int,
                 obstacles: Set[Position],
                 cost_model: Optional[EnergyAwareCost] = None,
                 max_time: int = 100,
                 max_nodes: int = 5000):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.cost_model = cost_model or EnergyAwareCost()
        self.max_time = max_time
        self.max_nodes = max_nodes
        self._node_counter = 0

    def solve(self, starts: List[Position], goals: List[Position]
              ) -> Optional[Dict[int, List[Position]]]:
        """
        Find conflict-free paths for all agents.

        Args:
            starts: list of (x, y) start positions, indexed by agent ID
            goals: list of (x, y) goal positions, indexed by agent ID

        Returns:
            Dict mapping agent_id -> list of (x, y) positions per timestep,
            or None if no solution found.
        """
        num_agents = len(starts)
        assert len(goals) == num_agents

        # Create root node with no constraints
        root = self._create_node()
        for agent_id in range(num_agents):
            root.constraints[agent_id] = (set(), set())

        # Compute initial paths (no constraints)
        congestion_map: Dict[Position, int] = {}
        for agent_id in range(num_agents):
            path = self._low_level_search(
                starts[agent_id], goals[agent_id],
                set(), set(), congestion_map)
            if path is None:
                return None  # No path possible for this agent
            root.solution[agent_id] = path
            # Update congestion map
            for pos in path:
                congestion_map[pos] = congestion_map.get(pos, 0) + 1

        root.cost = self._compute_solution_cost(root.solution)
        root.conflicts = self._find_all_conflicts(root.solution)

        # Priority queue (min-cost first)
        open_list = [root]
        nodes_expanded = 0

        while open_list and nodes_expanded < self.max_nodes:
            node = heapq.heappop(open_list)
            nodes_expanded += 1

            # No conflicts — we have a valid solution
            if not node.conflicts:
                return node.solution

            # Pick first conflict to resolve
            conflict = node.conflicts[0]

            # Branch: create two child nodes, one constraining each agent
            for agent_id in [conflict.agent_i, conflict.agent_j]:
                child = self._create_node()

                # Copy constraints from parent
                for aid in node.constraints:
                    v_set, e_set = node.constraints[aid]
                    child.constraints[aid] = (set(v_set), set(e_set))

                # Add new constraint for this agent
                if conflict.conflict_type == 'vertex':
                    new_vc = VertexConstraint(
                        conflict.x1, conflict.y1, conflict.timestep)
                    child.constraints[agent_id][0].add(new_vc)
                elif conflict.conflict_type == 'edge':
                    if agent_id == conflict.agent_i:
                        new_ec = EdgeConstraint(
                            conflict.x1, conflict.y1,
                            conflict.x2, conflict.y2,
                            conflict.timestep)
                    else:
                        new_ec = EdgeConstraint(
                            conflict.x2, conflict.y2,
                            conflict.x1, conflict.y1,
                            conflict.timestep)
                    child.constraints[agent_id][1].add(new_ec)

                # Copy solution from parent
                child.solution = dict(node.solution)

                # Re-plan only the constrained agent
                v_constraints, e_constraints = child.constraints[agent_id]

                # Build congestion map from other agents' paths
                cong = self._build_congestion_map(
                    child.solution, exclude_agent=agent_id)

                new_path = self._low_level_search(
                    starts[agent_id], goals[agent_id],
                    v_constraints, e_constraints, cong)

                if new_path is None:
                    continue  # This branch is dead

                child.solution[agent_id] = new_path
                child.cost = self._compute_solution_cost(child.solution)
                child.conflicts = self._find_all_conflicts(child.solution)

                heapq.heappush(open_list, child)

        # Exhausted search
        return None

    def _create_node(self) -> CTNode:
        """Create a new CT node with a unique ID."""
        self._node_counter += 1
        return CTNode(cost=0.0, id=self._node_counter)

    def _low_level_search(self, start: Position, goal: Position,
                          v_constraints: Set[VertexConstraint],
                          e_constraints: Set[EdgeConstraint],
                          congestion_map: Dict[Position, int]
                          ) -> Optional[List[Position]]:
        """Run Space-Time A* for a single agent."""
        astar = SpaceTimeAStar(
            self.grid_width, self.grid_height,
            self.obstacles, self.cost_model, self.max_time)
        return astar.search(start, goal, v_constraints, e_constraints,
                            congestion_map)

    def _compute_solution_cost(self, solution: Dict[int, List[Position]]
                               ) -> float:
        """Compute total cost of all paths (sum of individual path costs)."""
        total = 0.0
        for agent_id, path in solution.items():
            # Cost = path length (number of time steps)
            total += len(path)
        return total

    def _build_congestion_map(self, solution: Dict[int, List[Position]],
                              exclude_agent: int = -1
                              ) -> Dict[Position, int]:
        """Build a congestion map from all agents' paths (except excluded)."""
        cong: Dict[Position, int] = {}
        for agent_id, path in solution.items():
            if agent_id == exclude_agent:
                continue
            for pos in path:
                cong[pos] = cong.get(pos, 0) + 1
        return cong

    def _find_all_conflicts(self, solution: Dict[int, List[Position]]
                            ) -> List[Conflict]:
        """Detect all pairwise conflicts in the current solution."""
        conflicts = []
        agents = sorted(solution.keys())
        max_t = max(len(path) for path in solution.values()) if solution else 0

        for i, j in itertools.combinations(agents, 2):
            path_i = solution[i]
            path_j = solution[j]

            for t in range(max_t):
                # Get positions at time t (stay at goal if path ended)
                pos_i = path_i[t] if t < len(path_i) else path_i[-1]
                pos_j = path_j[t] if t < len(path_j) else path_j[-1]

                # Vertex conflict
                if pos_i == pos_j:
                    conflicts.append(Conflict(
                        agent_i=i, agent_j=j,
                        conflict_type='vertex',
                        x1=pos_i[0], y1=pos_i[1],
                        timestep=t))

                # Edge conflict (swap)
                if t + 1 < max_t:
                    next_i = path_i[t + 1] if t + 1 < len(path_i) else path_i[-1]
                    next_j = path_j[t + 1] if t + 1 < len(path_j) else path_j[-1]

                    if pos_i == next_j and pos_j == next_i:
                        conflicts.append(Conflict(
                            agent_i=i, agent_j=j,
                            conflict_type='edge',
                            x1=pos_i[0], y1=pos_i[1],
                            x2=pos_j[0], y2=pos_j[1],
                            timestep=t))

        return conflicts

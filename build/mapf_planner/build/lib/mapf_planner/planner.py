"""
Conflict-Based Search (CBS) Multi-Agent Path Finding planner with
Ant Colony Optimization (ACO) low-level search and energy-aware costs.

Subscribes to:
  /task_assignments  (std_msgs/String)  — JSON list of {robot, goal_x, goal_y}
  /robotN/odom       (nav_msgs/Odometry) — current pose of each robot

Publishes:
  /robotN/planned_path (nav_msgs/Path) — collision-free path for each robot
  /mapf_planner/stats  (std_msgs/String) — planning stats JSON
"""

import heapq
import json
import math
import random
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Grid-based ACO low-level search
# ---------------------------------------------------------------------------

class GridMap:
    """Simple occupancy grid for the warehouse."""

    def __init__(self, width=32, height=24, resolution=0.5, origin_x=-8.0, origin_y=-6.0):
        self.width = width          # cells
        self.height = height        # cells
        self.resolution = resolution
        self.origin_x = origin_x    # world x of cell (0,0)
        self.origin_y = origin_y    # world y of cell (0,0)
        self.obstacles = set()
        self._build_warehouse_obstacles()

    def _build_warehouse_obstacles(self):
        """Mark static structures as obstacles (matching smart_warehouse.world)."""
        shelf_positions = [
            # Row A  (y=3)
            (-5, 3), (-2, 3), (2, 3), (5, 3),
            # Row B  (y=0)
            (-5, 0), (-2, 0), (2, 0), (5, 0),
            # Row C  (y=-3)
            (-5, -3), (-2, -3), (2, -3), (5, -3),
        ]
        shelf_half_x = 1.0
        shelf_half_y = 0.25

        def mark_box(center_x, center_y, half_x, half_y):
            for dx in self._frange(center_x - half_x, center_x + half_x, self.resolution):
                for dy in self._frange(center_y - half_y, center_y + half_y, self.resolution):
                    gx, gy = self.world_to_grid(dx, dy)
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        self.obstacles.add((gx, gy))

        def mark_cylinder(center_x, center_y, radius):
            for dx in self._frange(center_x - radius, center_x + radius, self.resolution):
                for dy in self._frange(center_y - radius, center_y + radius, self.resolution):
                    if (dx - center_x) ** 2 + (dy - center_y) ** 2 > radius ** 2:
                        continue
                    gx, gy = self.world_to_grid(dx, dy)
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        self.obstacles.add((gx, gy))

        for sx, sy in shelf_positions:
            mark_box(sx, sy, shelf_half_x, shelf_half_y)

        # Cross-dock sorting island and mid-aisle dividers
        mark_box(0.0, -4.2, 1.5, 0.6)
        mark_box(-3.6, -1.6, 0.2, 1.3)
        mark_box(3.6, -1.6, 0.2, 1.3)

        # Safety pillars
        mark_cylinder(-6.0, 2.2, 0.28)
        mark_cylinder(6.0, 2.2, 0.28)

        # Staging pallets
        mark_box(-1.4, 4.5, 0.6, 0.45)
        mark_box(1.4, 4.5, 0.6, 0.45)

        # Perimeter walls
        for gx in range(self.width):
            self.obstacles.add((gx, 0))
            self.obstacles.add((gx, self.height - 1))
        for gy in range(self.height):
            self.obstacles.add((0, gy))
            self.obstacles.add((self.width - 1, gy))

    @staticmethod
    def _frange(start, stop, step):
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(v)
            v += step
        return vals

    def world_to_grid(self, wx, wy):
        gx = int(round((wx - self.origin_x) / self.resolution))
        gy = int(round((wy - self.origin_y) / self.resolution))
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin_x
        wy = gy * self.resolution + self.origin_y
        return (wx, wy)

    def is_free(self, gx, gy):
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return (gx, gy) not in self.obstacles

    def neighbors(self, gx, gy):
        """4-connected + wait-in-place."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]:
            nx, ny = gx + dx, gy + dy
            if self.is_free(nx, ny):
                yield (nx, ny)


# ---------------------------------------------------------------------------
# Space-Time ACO
# ---------------------------------------------------------------------------

def spacetime_aco(
    grid_map,
    start,
    goal,
    constraints,
    energy_weight=0.3,
    ant_count=30,
    aco_iterations=25,
    alpha=1.0,
    beta=2.5,
    evaporation=0.2,
    deposit_scale=40.0,
    max_t=200,
):
    """
    Ant Colony Optimization in (x, y, t) space with constraints.
    constraints: set of ((x, y, t)) — vertex constraints
                 set of ((x1, y1, x2, y2, t)) — edge constraints
    energy_weight: penalty multiplier for distance (models energy cost)
    """
    vertex_constraints = {c for c in constraints if len(c) == 3}
    edge_constraints = {c for c in constraints if len(c) == 5}

    def heuristic(pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def transition_cost(prev_state, next_state):
        move_cost = 1.0
        if (prev_state[0], prev_state[1]) != (next_state[0], next_state[1]):
            move_cost += energy_weight
        return move_cost

    def state_neighbors(state):
        cx, cy, ct = state
        if ct >= max_t:
            return []

        neighbors = []
        nt = ct + 1
        for nx, ny in grid_map.neighbors(cx, cy):
            if (nx, ny, nt) in vertex_constraints:
                continue
            if (cx, cy, nx, ny, nt) in edge_constraints:
                continue
            neighbors.append((nx, ny, nt))
        return neighbors

    def path_cost(state_path):
        total = 0.0
        for i in range(1, len(state_path)):
            total += transition_cost(state_path[i - 1], state_path[i])
        return total

    start_state = (start[0], start[1], 0)
    pheromone = {}

    best_state_path = None
    best_cost = float('inf')

    for _ in range(aco_iterations):
        successful_paths = []

        for _ in range(ant_count):
            current = start_state
            ant_path = [current]
            ant_visited = {current}

            while current[2] < max_t:
                if (current[0], current[1]) == goal and current[2] > 0:
                    break

                nbrs = state_neighbors(current)
                if not nbrs:
                    break

                weights = []
                for nxt in nbrs:
                    edge = (current, nxt)
                    tau = pheromone.get(edge, 1.0) ** alpha
                    eta = (1.0 / (heuristic((nxt[0], nxt[1]), goal) + 1.0)) ** beta

                    if nxt in ant_visited:
                        weight = tau * eta * 0.15
                    else:
                        weight = tau * eta

                    weights.append((nxt, weight))

                total_w = sum(w for _, w in weights)
                if total_w <= 1e-9:
                    break

                pick = random.random() * total_w
                cumulative = 0.0
                chosen = weights[-1][0]
                for nxt, weight in weights:
                    cumulative += weight
                    if cumulative >= pick:
                        chosen = nxt
                        break

                ant_path.append(chosen)
                ant_visited.add(chosen)
                current = chosen

            if (ant_path[-1][0], ant_path[-1][1]) == goal and ant_path[-1][2] > 0:
                cost = path_cost(ant_path)
                successful_paths.append((ant_path, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_state_path = ant_path

        for edge in list(pheromone.keys()):
            pheromone[edge] *= (1.0 - evaporation)
            if pheromone[edge] < 1e-6:
                pheromone[edge] = 1e-6

        for ant_path, cost in successful_paths:
            deposit = deposit_scale / max(cost, 1e-6)
            for i in range(1, len(ant_path)):
                edge = (ant_path[i - 1], ant_path[i])
                pheromone[edge] = pheromone.get(edge, 1.0) + deposit

    if best_state_path is None:
        return None

    return [(sx, sy) for sx, sy, _ in best_state_path]


# ---------------------------------------------------------------------------
# CBS (Conflict-Based Search)
# ---------------------------------------------------------------------------

def detect_conflicts(paths):
    """Detect first vertex or edge conflict among agent paths."""
    agents = list(paths.keys())
    max_t = max(len(p) for p in paths.values()) if paths else 0

    for t in range(max_t):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                pi = paths[ai][min(t, len(paths[ai]) - 1)]
                pj = paths[aj][min(t, len(paths[aj]) - 1)]

                # Vertex conflict
                if pi == pj:
                    return {'type': 'vertex', 'agents': (ai, aj),
                            'position': pi, 'time': t}

                # Edge (swap) conflict
                if t > 0:
                    pi_prev = paths[ai][min(t - 1, len(paths[ai]) - 1)]
                    pj_prev = paths[aj][min(t - 1, len(paths[aj]) - 1)]
                    if pi == pj_prev and pj == pi_prev:
                        return {'type': 'edge', 'agents': (ai, aj),
                                'positions': (pi_prev, pi), 'time': t}
    return None


class CBSNode:
    """Node in the CBS search tree."""
    def __init__(self):
        self.constraints = {}   # agent -> set of constraints
        self.paths = {}         # agent -> list of (gx, gy)
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost


def cbs_search(
    grid_map,
    starts,
    goals,
    energy_weight=0.3,
    ant_count=30,
    aco_iterations=25,
    aco_alpha=1.0,
    aco_beta=2.5,
    aco_evaporation=0.2,
    aco_deposit_scale=40.0,
):
    """
    Conflict-Based Search for multi-agent path finding.
    starts: dict  agent_name -> (gx, gy)
    goals:  dict  agent_name -> (gx, gy)
    Returns dict  agent_name -> [(gx, gy), ...]
    """
    root = CBSNode()
    root.constraints = {agent: set() for agent in starts}

    # Initial paths (no constraints)
    for agent in starts:
        path = spacetime_aco(
            grid_map,
            starts[agent],
            goals[agent],
            set(),
            energy_weight=energy_weight,
            ant_count=ant_count,
            aco_iterations=aco_iterations,
            alpha=aco_alpha,
            beta=aco_beta,
            evaporation=aco_evaporation,
            deposit_scale=aco_deposit_scale,
        )
        if path is None:
            return None
        root.paths[agent] = path

    root.cost = sum(len(p) for p in root.paths.values())

    open_set = [root]

    iterations = 0
    while open_set and iterations < 500:
        iterations += 1
        node = heapq.heappop(open_set)

        conflict = detect_conflicts(node.paths)
        if conflict is None:
            return node.paths  # Solution found

        # Branch on conflict
        for agent in conflict['agents']:
            child = CBSNode()
            child.constraints = {a: set(c) for a, c in node.constraints.items()}

            if conflict['type'] == 'vertex':
                pos = conflict['position']
                t = conflict['time']
                child.constraints[agent].add((pos[0], pos[1], t))
            else:  # edge
                p1, p2 = conflict['positions']
                t = conflict['time']
                child.constraints[agent].add((p1[0], p1[1], p2[0], p2[1], t))

            child.paths = dict(node.paths)

            # Re-plan for constrained agent
            new_path = spacetime_aco(
                grid_map,
                starts[agent],
                goals[agent],
                child.constraints[agent],
                energy_weight=energy_weight,
                ant_count=ant_count,
                aco_iterations=aco_iterations,
                alpha=aco_alpha,
                beta=aco_beta,
                evaporation=aco_evaporation,
                deposit_scale=aco_deposit_scale,
            )
            if new_path is None:
                continue

            child.paths[agent] = new_path
            child.cost = sum(len(p) for p in child.paths.values())
            heapq.heappush(open_set, child)

    return None  # Failed


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class MAPFPlanner(Node):
    def __init__(self):
        super().__init__('mapf_planner')

        self.declare_parameter('num_robots', 3)
        self.declare_parameter('energy_weight', 0.3)
        self.declare_parameter('aco_ant_count', 30)
        self.declare_parameter('aco_iterations', 25)
        self.declare_parameter('aco_alpha', 1.0)
        self.declare_parameter('aco_beta', 2.5)
        self.declare_parameter('aco_evaporation', 0.2)
        self.declare_parameter('aco_deposit_scale', 40.0)

        self.num_robots = self.get_parameter('num_robots').value
        self.energy_weight = self.get_parameter('energy_weight').value
        self.aco_ant_count = self.get_parameter('aco_ant_count').value
        self.aco_iterations = self.get_parameter('aco_iterations').value
        self.aco_alpha = self.get_parameter('aco_alpha').value
        self.aco_beta = self.get_parameter('aco_beta').value
        self.aco_evaporation = self.get_parameter('aco_evaporation').value
        self.aco_deposit_scale = self.get_parameter('aco_deposit_scale').value

        self.grid_map = GridMap()

        # Robot names
        self.robot_names = [f'robot{i+1}' for i in range(self.num_robots)]

        # Current poses (world coords)
        self.current_poses = {}

        # Publishers per robot
        self.path_pubs = {}
        for name in self.robot_names:
            self.path_pubs[name] = self.create_publisher(
                Path, f'/{name}/planned_path', 10)
            self.create_subscription(
                Odometry, f'/{name}/odom',
                lambda msg, n=name: self._odom_cb(msg, n), 10)

        # Task assignment subscriber
        self.create_subscription(
            String, '/task_assignments', self._task_cb, 10)

        # Stats publisher
        self.stats_pub = self.create_publisher(String, '/mapf_planner/stats', 10)

        self.get_logger().info(
            f'CBS+ACO MAPF Planner started — {self.num_robots} robots, '
            f'energy_weight={self.energy_weight}, ants={self.aco_ant_count}, '
            f'iters={self.aco_iterations}')

    # ---- callbacks --------------------------------------------------------

    def _odom_cb(self, msg, robot_name):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_poses[robot_name] = (x, y)

    def _task_cb(self, msg):
        """
        Expects JSON: [{"robot": "robot1", "goal_x": 2.0, "goal_y": 3.0}, ...]
        """
        try:
            assignments = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in /task_assignments')
            return

        starts = {}
        goals = {}
        for a in assignments:
            name = a['robot']
            if name not in self.current_poses:
                self.get_logger().warn(f'No odom for {name}, using spawn default')
                self.current_poses[name] = (-6.5, 4.5)

            wx, wy = self.current_poses[name]
            starts[name] = self.grid_map.world_to_grid(wx, wy)
            goals[name] = self.grid_map.world_to_grid(a['goal_x'], a['goal_y'])

        self.get_logger().info(f'Planning paths for {list(starts.keys())}...')
        t0 = time.time()

        solution = cbs_search(
            self.grid_map,
            starts,
            goals,
            energy_weight=self.energy_weight,
            ant_count=self.aco_ant_count,
            aco_iterations=self.aco_iterations,
            aco_alpha=self.aco_alpha,
            aco_beta=self.aco_beta,
            aco_evaporation=self.aco_evaporation,
            aco_deposit_scale=self.aco_deposit_scale,
        )

        elapsed = time.time() - t0

        if solution is None:
            self.get_logger().error('CBS+ACO failed to find solution')
            return

        # Publish paths
        total_cost = 0
        total_energy = 0.0
        for name, grid_path in solution.items():
            ros_path = Path()
            ros_path.header.frame_id = 'map'
            ros_path.header.stamp = self.get_clock().now().to_msg()

            energy = 0.0
            for k, (gx, gy) in enumerate(grid_path):
                wx, wy = self.grid_map.grid_to_world(gx, gy)
                ps = PoseStamped()
                ps.header.frame_id = 'map'
                ps.pose.position.x = wx
                ps.pose.position.y = wy
                ps.pose.position.z = 0.0
                ros_path.poses.append(ps)

                if k > 0:
                    prev = grid_path[k - 1]
                    if (gx, gy) != prev:
                        energy += 1.0 + self.energy_weight

            self.path_pubs[name].publish(ros_path)
            total_cost += len(grid_path)
            total_energy += energy
            self.get_logger().info(
                f'  {name}: {len(grid_path)} steps, energy={energy:.1f}')

        # Publish stats
        stats = {
            'planning_time_ms': round(elapsed * 1000, 1),
            'total_cost': total_cost,
            'total_energy': round(total_energy, 1),
            'num_agents': len(solution),
            'paths': {n: len(p) for n, p in solution.items()},
        }
        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self.stats_pub.publish(stats_msg)

        self.get_logger().info(
            f'CBS+ACO solved in {elapsed*1000:.1f}ms — '
            f'total_cost={total_cost}, energy={total_energy:.1f}')


def main(args=None):
    rclpy.init(args=args)
    node = MAPFPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

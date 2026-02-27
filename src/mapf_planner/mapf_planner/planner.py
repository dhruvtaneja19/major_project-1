"""
Energy-Aware Distributed Swarm Coordination Planner - ROS 2 Node.

Implements Space-Time Multi-Agent Path Planning using CBS
(Conflict-Based Search) with an energy-aware cost model.

Subscribes:
  /task_assignments      (std_msgs/String)  - JSON array of robot-goal pairs
  /robotN/odom           (nav_msgs/Odometry) - each robot's position
  /robotN/battery_level  (std_msgs/Float32)  - each robot's battery

Publishes:
  /robotN/planned_path   (nav_msgs/Path)     - conflict-free path per robot
  /mapf_planner/stats    (std_msgs/String)    - planning statistics JSON

The planner converts continuous arena coordinates to a discrete grid,
runs CBS to produce conflict-free space-time paths, then converts back
to continuous waypoints and publishes nav_msgs/Path for each robot.

Energy-aware cost model:
  Cost = w_dist * distance + w_turn * turns + w_wait * idle + w_cong * congestion
"""

import json
import math
import time
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String

from .cbs import CBS
from .space_time_astar import EnergyAwareCost, Position

# ── Arena Grid Configuration ──────────────────────────────────────────
# The arena is 16m x 12m, centered at origin.
# Grid resolution: 0.5m per cell -> 32 x 24 grid
GRID_RESOLUTION = 0.5  # meters per cell
WORLD_X_MIN = -8.0
WORLD_Y_MIN = -6.0
WORLD_X_MAX = 8.0
WORLD_Y_MAX = 6.0
GRID_WIDTH = int((WORLD_X_MAX - WORLD_X_MIN) / GRID_RESOLUTION)   # 32
GRID_HEIGHT = int((WORLD_Y_MAX - WORLD_Y_MIN) / GRID_RESOLUTION)  # 24

# Static obstacle cells -- computed from world model
# 6 pillar obstacles (1.5m x 1.5m boxes) + 1 central barrier (4.0m x 0.3m)
PILLAR_CENTERS = [
    (-4, 3), (0, 3), (4, 3),
    (-4, -3), (0, -3), (4, -3),
]
PILLAR_HALF = 0.75  # half-width of pillar (1.5m / 2)

BARRIER_CENTER = (0, 0)
BARRIER_HALF_W = 2.0  # half of 4.0m
BARRIER_HALF_H = 0.15  # half of 0.3m


def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
    """Convert world coordinates to grid cell indices."""
    gx = int((wx - WORLD_X_MIN) / GRID_RESOLUTION)
    gy = int((wy - WORLD_Y_MIN) / GRID_RESOLUTION)
    gx = max(0, min(GRID_WIDTH - 1, gx))
    gy = max(0, min(GRID_HEIGHT - 1, gy))
    return (gx, gy)


def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    """Convert grid cell indices back to world coordinates (cell center)."""
    wx = WORLD_X_MIN + (gx + 0.5) * GRID_RESOLUTION
    wy = WORLD_Y_MIN + (gy + 0.5) * GRID_RESOLUTION
    return (wx, wy)


def build_obstacle_set() -> Set[Position]:
    """Build set of grid cells occupied by static obstacles (pillars + barrier + walls)."""
    obstacles = set()

    # Pillar obstacles (1.5m x 1.5m square blocks)
    for cx, cy in PILLAR_CENTERS:
        for dx_i in range(-int(PILLAR_HALF / GRID_RESOLUTION),
                          int(PILLAR_HALF / GRID_RESOLUTION) + 1):
            for dy_i in range(-int(PILLAR_HALF / GRID_RESOLUTION),
                              int(PILLAR_HALF / GRID_RESOLUTION) + 1):
                wx = cx + dx_i * GRID_RESOLUTION
                wy = cy + dy_i * GRID_RESOLUTION
                gx, gy = world_to_grid(wx, wy)
                obstacles.add((gx, gy))

    # Central barrier (4.0m x 0.3m)
    bcx, bcy = BARRIER_CENTER
    for dx_i in range(-int(BARRIER_HALF_W / GRID_RESOLUTION),
                      int(BARRIER_HALF_W / GRID_RESOLUTION) + 1):
        for dy_i in range(-int(BARRIER_HALF_H / GRID_RESOLUTION),
                          int(BARRIER_HALF_H / GRID_RESOLUTION) + 1):
            wx = bcx + dx_i * GRID_RESOLUTION
            wy = bcy + dy_i * GRID_RESOLUTION
            gx, gy = world_to_grid(wx, wy)
            obstacles.add((gx, gy))

    # Perimeter walls (1 cell thick)
    for gx in range(GRID_WIDTH):
        obstacles.add((gx, 0))
        obstacles.add((gx, GRID_HEIGHT - 1))
    for gy in range(GRID_HEIGHT):
        obstacles.add((0, gy))
        obstacles.add((GRID_WIDTH - 1, gy))

    return obstacles


class MAPFPlannerNode(Node):
    """
    ROS 2 node implementing CBS-based multi-agent space-time planning
    with energy-aware cost optimization.
    """

    def __init__(self):
        super().__init__('mapf_planner')

        # ── Parameters ──
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('energy_weight_distance', 1.0)
        self.declare_parameter('energy_weight_turns', 0.3)
        self.declare_parameter('energy_weight_wait', 0.2)
        self.declare_parameter('energy_weight_congestion', 0.4)
        self.declare_parameter('max_planning_time_steps', 120)
        self.declare_parameter('max_cbs_nodes', 5000)
        self.declare_parameter('grid_resolution', 0.5)

        self.num_robots = self.get_parameter('num_robots').value
        w_dist = self.get_parameter('energy_weight_distance').value
        w_turn = self.get_parameter('energy_weight_turns').value
        w_wait = self.get_parameter('energy_weight_wait').value
        w_cong = self.get_parameter('energy_weight_congestion').value
        max_t = self.get_parameter('max_planning_time_steps').value
        max_nodes = self.get_parameter('max_cbs_nodes').value

        # ── Energy-Aware Cost Model ──
        self.cost_model = EnergyAwareCost(
            w_dist=w_dist, w_turn=w_turn,
            w_wait=w_wait, w_cong=w_cong)

        # ── Grid & Obstacles ──
        self.obstacles = build_obstacle_set()
        self.get_logger().info(
            f'Grid: {GRID_WIDTH}x{GRID_HEIGHT}, '
            f'{len(self.obstacles)} obstacle cells')

        # ── CBS Solver ──
        self.cbs = CBS(
            grid_width=GRID_WIDTH,
            grid_height=GRID_HEIGHT,
            obstacles=self.obstacles,
            cost_model=self.cost_model,
            max_time=max_t,
            max_nodes=max_nodes)

        # ── Robot State ──
        self.robot_positions: Dict[str, Tuple[float, float]] = {}
        self.robot_battery: Dict[str, float] = {}
        self.robot_names = [f'robot{i + 1}' for i in range(self.num_robots)]

        # ── Publishers ──
        self.path_pubs: Dict[str, object] = {}
        for name in self.robot_names:
            self.path_pubs[name] = self.create_publisher(
                Path, f'/{name}/planned_path', 10)

        self.stats_pub = self.create_publisher(
            String, '/mapf_planner/stats', 10)

        # ── Subscribers ──
        for name in self.robot_names:
            self.create_subscription(
                Odometry, f'/{name}/odom',
                lambda msg, n=name: self._odom_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{name}/battery_level',
                lambda msg, n=name: self._battery_cb(msg, n), 10)

        self.create_subscription(
            String, '/task_assignments', self._task_cb, 10)

        self.get_logger().info(
            f'MAPF Planner (CBS + Space-Time A*) started — '
            f'{self.num_robots} robots, '
            f'cost weights: dist={w_dist}, turn={w_turn}, '
            f'wait={w_wait}, cong={w_cong}')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg, name: str):
        self.robot_positions[name] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y)

    def _battery_cb(self, msg, name: str):
        self.robot_battery[name] = msg.data

    def _task_cb(self, msg):
        """
        Receive task assignments, compute conflict-free paths using CBS,
        and publish planned paths for each robot.
        """
        try:
            assignments = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task_assignments')
            return

        if not assignments:
            return

        self.get_logger().info(
            f'Received {len(assignments)} task assignments — planning...')

        # Build agent start/goal lists
        agent_starts: List[Position] = []
        agent_goals: List[Position] = []
        agent_names: List[str] = []

        for assignment in assignments:
            robot_name = assignment['robot']
            goal_x = assignment['goal_x']
            goal_y = assignment['goal_y']

            # Get current position from odometry (or default)
            if robot_name in self.robot_positions:
                wx, wy = self.robot_positions[robot_name]
            else:
                self.get_logger().warn(
                    f'No odom for {robot_name}, using (0,0)')
                wx, wy = 0.0, 0.0

            start_grid = world_to_grid(wx, wy)
            goal_grid = world_to_grid(goal_x, goal_y)

            # Ensure start/goal are not on obstacles
            start_grid = self._nearest_free_cell(start_grid)
            goal_grid = self._nearest_free_cell(goal_grid)

            agent_starts.append(start_grid)
            agent_goals.append(goal_grid)
            agent_names.append(robot_name)

        # ── Run CBS ──
        t_start = time.time()
        solution = self.cbs.solve(agent_starts, agent_goals)
        t_elapsed = (time.time() - t_start) * 1000  # ms

        # ── Publish stats ──
        stats = {
            'planning_time_ms': round(t_elapsed, 2),
            'num_agents': len(agent_names),
            'success': solution is not None,
            'cost_weights': {
                'distance': self.cost_model.w_dist,
                'turns': self.cost_model.w_turn,
                'wait': self.cost_model.w_wait,
                'congestion': self.cost_model.w_cong,
            },
        }

        if solution:
            stats['total_path_steps'] = sum(
                len(p) for p in solution.values())
            stats['max_makespan'] = max(
                len(p) for p in solution.values())

            # Count turns across all paths
            total_turns = 0
            total_waits = 0
            for path in solution.values():
                for i in range(1, len(path)):
                    if path[i] == path[i - 1]:
                        total_waits += 1
                    elif i >= 2:
                        dx_prev = path[i - 1][0] - path[i - 2][0]
                        dy_prev = path[i - 1][1] - path[i - 2][1]
                        dx_curr = path[i][0] - path[i - 1][0]
                        dy_curr = path[i][1] - path[i - 1][1]
                        if (dx_prev, dy_prev) != (dx_curr, dy_curr):
                            total_turns += 1
            stats['total_turns'] = total_turns
            stats['total_waits'] = total_waits

        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self.stats_pub.publish(stats_msg)

        if solution is None:
            self.get_logger().error(
                f'CBS FAILED to find solution in {t_elapsed:.1f}ms')
            return

        self.get_logger().info(
            f'CBS solved in {t_elapsed:.1f}ms — '
            f'makespan={stats["max_makespan"]}, '
            f'turns={stats.get("total_turns", 0)}, '
            f'waits={stats.get("total_waits", 0)}')

        # ── Publish paths ──
        for agent_id, robot_name in enumerate(agent_names):
            grid_path = solution[agent_id]
            self._publish_path(robot_name, grid_path)

    def _publish_path(self, robot_name: str, grid_path: List[Position]):
        """Convert grid path to nav_msgs/Path and publish."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'

        for gx, gy in grid_path:
            wx, wy = grid_to_world(gx, gy)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.path_pubs[robot_name].publish(path_msg)
        self.get_logger().info(
            f'Published path for {robot_name}: '
            f'{len(grid_path)} waypoints')

    def _nearest_free_cell(self, pos: Position,
                           max_radius: int = 5) -> Position:
        """Find nearest obstacle-free cell using BFS spiral."""
        if pos not in self.obstacles:
            return pos
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                            (nx, ny) not in self.obstacles):
                        return (nx, ny)
        return pos  # fallback


def main(args=None):
    rclpy.init(args=args)
    node = MAPFPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

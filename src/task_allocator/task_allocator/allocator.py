"""
Energy-Aware Distributed Swarm Task Allocator.

Coordinates autonomous mobile robots in an arena environment using
energy-distance-congestion heuristics. Supports 3-5 robot swarms with:

  - Energy-aware task assignment (battery + distance + congestion scoring)
  - Automatic low-battery charging dispatch
  - Decentralized goal execution via CBS planner
  - Cooperative neighbor awareness for load balancing
  - Swarm scalability (parameterized robot count)

Publishes:
  /task_assignments        (std_msgs/String) - JSON for CBS planner
  /task_allocator/status   (std_msgs/String) - swarm status JSON

Subscribes:
  /robotN/odom             (nav_msgs/Odometry)
  /robotN/battery_level    (std_msgs/Float32)
  /robotN/status           (std_msgs/String) - from path followers

Part of: Energy-Aware Distributed Swarm Coordination of Autonomous
         Mobile Robots Using Space-Time Multi-Agent Path Planning
"""

import json
import math
import time
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32


# ── Arena Waypoint Locations (world coordinates) ─────────────────────
WAYPOINT_LOCATIONS = [
    # Upper-left zone
    {'name': 'W1', 'x': -6.0, 'y': 4.5},
    {'name': 'W2', 'x': -2.0, 'y': 4.5},
    # Upper-right zone
    {'name': 'W3', 'x': 2.0, 'y': 4.5},
    {'name': 'W4', 'x': 6.0, 'y': 4.5},
    # Mid-left (above barrier)
    {'name': 'W5', 'x': -6.0, 'y': 1.5},
    {'name': 'W6', 'x': -2.0, 'y': 1.5},
    # Mid-right (above barrier)
    {'name': 'W7', 'x': 2.0, 'y': 1.5},
    {'name': 'W8', 'x': 6.0, 'y': 1.5},
    # Lower-left (below barrier)
    {'name': 'W9', 'x': -6.0, 'y': -1.5},
    {'name': 'W10', 'x': -2.0, 'y': -1.5},
    # Lower-right
    {'name': 'W11', 'x': 2.0, 'y': -1.5},
    {'name': 'W12', 'x': 6.0, 'y': -1.5},
    # Bottom-left
    {'name': 'W13', 'x': -6.0, 'y': -4.5},
    {'name': 'W14', 'x': -2.0, 'y': -4.5},
    # Bottom-right
    {'name': 'W15', 'x': 2.0, 'y': -4.5},
    {'name': 'W16', 'x': 6.0, 'y': -4.5},
]

REST_ZONE = {'name': 'RestZone', 'x': -6.0, 'y': -4.5}
CHARGING_STATION = {'name': 'ChargingStation', 'x': 6.0, 'y': -4.5}

LOW_BATTERY_THRESHOLD = 25.0     # send to charger below this %
CRITICAL_BATTERY = 10.0          # refuse new tasks


class TaskAllocator(Node):

    def __init__(self):
        super().__init__('task_allocator')

        # ── Parameters (swarm-scalable) ──
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('task_interval', 15.0)
        self.declare_parameter('energy_weight', 0.4)
        self.declare_parameter('distance_weight', 0.4)
        self.declare_parameter('congestion_weight', 0.2)

        self.num_robots = self.get_parameter('num_robots').value
        self.task_interval = self.get_parameter('task_interval').value
        self.w_energy = self.get_parameter('energy_weight').value
        self.w_dist = self.get_parameter('distance_weight').value
        self.w_cong = self.get_parameter('congestion_weight').value

        self.robot_names = [f'robot{i + 1}' for i in range(self.num_robots)]

        # ── State tracking ──
        self.robot_poses = {}       # name -> (x, y)
        self.robot_battery = {}     # name -> float
        self.robot_busy = {}        # name -> bool
        self.robot_active = {}      # name -> bool (from path follower status)
        self.task_queue = []
        self.completed_tasks = 0
        self.total_assignments = 0
        self.congestion_zones = {}  # (grid_region) -> count of recent visits

        # ── Subscribers ──
        for name in self.robot_names:
            self.create_subscription(
                Odometry, f'/{name}/odom',
                lambda msg, n=name: self._odom_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{name}/battery_level',
                lambda msg, n=name: self._battery_cb(msg, n), 10)
            self.create_subscription(
                String, f'/{name}/status',
                lambda msg, n=name: self._robot_status_cb(msg, n), 10)

        # ── Publishers ──
        self.task_pub = self.create_publisher(String, '/task_assignments', 10)
        self.status_pub = self.create_publisher(
            String, '/task_allocator/status', 10)

        # ── Timers ──
        self.task_timer = self.create_timer(
            self.task_interval, self._generate_and_assign)
        self.status_timer = self.create_timer(5.0, self._publish_status)

        # ── Initialize defaults ──
        for name in self.robot_names:
            self.robot_battery[name] = 100.0
            self.robot_busy[name] = False
            self.robot_active[name] = False

        self.get_logger().info(
            f'Swarm TaskAllocator started — {self.num_robots} robots, '
            f'interval={self.task_interval}s, '
            f'weights: energy={self.w_energy}, dist={self.w_dist}, '
            f'cong={self.w_cong}')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg, robot_name):
        self.robot_poses[robot_name] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y)

    def _battery_cb(self, msg, robot_name):
        self.robot_battery[robot_name] = msg.data

    def _robot_status_cb(self, msg, robot_name):
        """Track whether robot is currently executing a path."""
        try:
            status = json.loads(msg.data)
            self.robot_active[robot_name] = status.get('active', False)
            # Auto-detect task completion
            if (not status.get('active', False) and
                    self.robot_busy.get(robot_name, False)):
                self.robot_busy[robot_name] = False
                self.completed_tasks += 1
        except json.JSONDecodeError:
            pass

    # ── Task Generation & Assignment ──────────────────────────────────────

    def _generate_and_assign(self):
        """
        Generate tasks and assign to available robots using
        energy-distance-congestion heuristic for swarm coordination.
        """
        # ── Phase 1: Low-battery robots → charging station ──
        charging_assignments = []
        for name in self.robot_names:
            bat = self.robot_battery.get(name, 100.0)
            if bat < LOW_BATTERY_THRESHOLD and not self.robot_busy.get(name):
                charging_assignments.append({
                    'robot': name,
                    'goal_x': CHARGING_STATION['x'],
                    'goal_y': CHARGING_STATION['y'],
                    'task_type': 'charge',
                })
                self.robot_busy[name] = True
                self.get_logger().warn(
                    f'[SWARM] {name} battery={bat:.1f}% → charging station')

        # ── Phase 2: Available robots for new tasks ──
        available = [
            n for n in self.robot_names
            if not self.robot_busy.get(n, False)
            and self.robot_battery.get(n, 100.0) >= CRITICAL_BATTERY
            and n in self.robot_poses
        ]

        if not available:
            if charging_assignments:
                self._publish_assignments(charging_assignments)
            return

        # Generate tasks (simulate incoming goal assignments)
        num_tasks = min(len(available), random.randint(1, min(3, len(available))))
        waypoints = random.sample(WAYPOINT_LOCATIONS, num_tasks)

        # ── Phase 3: Energy-distance-congestion assignment ──
        assignments = list(charging_assignments)
        used_robots = set(a['robot'] for a in charging_assignments)

        for waypoint in waypoints:
            best_robot = None
            best_score = float('inf')

            for name in available:
                if name in used_robots:
                    continue

                rx, ry = self.robot_poses.get(name, (0, 0))
                dist = math.hypot(waypoint['x'] - rx, waypoint['y'] - ry)
                bat = self.robot_battery.get(name, 100.0)

                # Congestion: count nearby robots heading to same region
                cong = self._compute_congestion(waypoint['x'], waypoint['y'], name)

                # Energy-aware score (lower is better):
                #   Score = w_dist * normalized_distance
                #         + w_energy * (100 - battery) / 100
                #         + w_cong * congestion_count
                max_dist = 20.0  # max possible warehouse distance
                score = (self.w_dist * (dist / max_dist) +
                         self.w_energy * ((100.0 - bat) / 100.0) +
                         self.w_cong * cong)

                if score < best_score:
                    best_score = score
                    best_robot = name

            if best_robot:
                assignments.append({
                    'robot': best_robot,
                    'goal_x': waypoint['x'],
                    'goal_y': waypoint['y'],
                    'task_type': 'navigate',
                    'waypoint': waypoint['name'],
                })
                used_robots.add(best_robot)
                self.robot_busy[best_robot] = True
                self.total_assignments += 1

                # Update congestion tracking
                region = self._region_key(waypoint['x'], waypoint['y'])
                self.congestion_zones[region] = \
                    self.congestion_zones.get(region, 0) + 1

                self.get_logger().info(
                    f'[SWARM] {best_robot} -> waypoint {waypoint["name"]} '
                    f'(bat={self.robot_battery.get(best_robot, 0):.0f}%, '
                    f'score={best_score:.3f})')

        if assignments:
            self._publish_assignments(assignments)

    def _compute_congestion(self, goal_x, goal_y, exclude_robot):
        """Estimate congestion near a goal based on other robots' positions."""
        congestion = 0
        region = self._region_key(goal_x, goal_y)

        # Count robots already near this area
        for name in self.robot_names:
            if name == exclude_robot:
                continue
            if name in self.robot_poses:
                rx, ry = self.robot_poses[name]
                if math.hypot(goal_x - rx, goal_y - ry) < 3.0:
                    congestion += 1

        # Historical congestion
        congestion += self.congestion_zones.get(region, 0) * 0.1
        return congestion

    def _region_key(self, x, y):
        """Discretize position to a region for congestion tracking."""
        return (round(x / 3.0), round(y / 3.0))

    def _publish_assignments(self, assignments):
        """Publish task assignments for the CBS space-time planner."""
        msg = String()
        msg.data = json.dumps(assignments)
        self.task_pub.publish(msg)
        self.get_logger().info(
            f'[SWARM] Published {len(assignments)} assignments to CBS planner')

    def _publish_status(self):
        """Publish swarm coordination status."""
        status = {
            'timestamp': time.time(),
            'num_robots': self.num_robots,
            'robots': {},
            'completed_tasks': self.completed_tasks,
            'total_assignments': self.total_assignments,
            'pending_tasks': len(self.task_queue),
            'swarm_health': self._compute_swarm_health(),
        }
        for name in self.robot_names:
            status['robots'][name] = {
                'battery': round(self.robot_battery.get(name, 0), 1),
                'position': list(self.robot_poses.get(name, (0, 0))),
                'busy': self.robot_busy.get(name, False),
                'active': self.robot_active.get(name, False),
            }

        # Reset busy flags for robots that finished (simple timeout approach)
        for name in self.robot_names:
            if (self.robot_busy.get(name, False) and
                    not self.robot_active.get(name, False)):
                self.robot_busy[name] = False

        # Decay congestion zones
        for region in list(self.congestion_zones.keys()):
            self.congestion_zones[region] *= 0.9
            if self.congestion_zones[region] < 0.1:
                del self.congestion_zones[region]

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _compute_swarm_health(self):
        """Compute overall swarm health score (0-100)."""
        if not self.robot_battery:
            return 100.0
        avg_bat = sum(self.robot_battery.values()) / len(self.robot_battery)
        active_ratio = sum(
            1 for n in self.robot_names
            if self.robot_battery.get(n, 0) > CRITICAL_BATTERY
        ) / self.num_robots
        return round(avg_bat * 0.6 + active_ratio * 40.0, 1)


def main(args=None):
    rclpy.init(args=args)
    node = TaskAllocator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

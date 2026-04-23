"""
Energy-Aware Task Allocator for warehouse robots.

Monitors battery levels, generates pickup/delivery tasks, and assigns
them to robots using a greedy energy-distance heuristic. Sends low-
battery robots to the charging station automatically.

Publishes:
  /task_assignments  (std_msgs/String) — JSON task assignments
  /task_allocator/status (std_msgs/String) — system status JSON

Subscribes:
  /robotN/odom          (nav_msgs/Odometry)
  /robotN/battery_level (std_msgs/Float32)
"""

import json
import math
import time
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32


# Warehouse locations (world coordinates)
SHELF_LOCATIONS = [
    # Row A
    {'name': 'A1', 'x': -5.0, 'y': 4.5},
    {'name': 'A2', 'x': -2.0, 'y': 4.5},
    {'name': 'A3', 'x': 2.0, 'y': 4.5},
    {'name': 'A4', 'x': 5.0, 'y': 4.5},
    # Row B
    {'name': 'B1', 'x': -5.0, 'y': 1.5},
    {'name': 'B2', 'x': -2.0, 'y': 1.5},
    {'name': 'B3', 'x': 2.0, 'y': 1.5},
    {'name': 'B4', 'x': 5.0, 'y': 1.5},
    # Row C
    {'name': 'C1', 'x': -5.0, 'y': -1.5},
    {'name': 'C2', 'x': -2.0, 'y': -1.5},
    {'name': 'C3', 'x': 2.0, 'y': -1.5},
    {'name': 'C4', 'x': 5.0, 'y': -1.5},
]

LOADING_DOCK = {'name': 'LoadingDock', 'x': -6.5, 'y': -4.5}
CHARGING_STATION = {'name': 'ChargingStation', 'x': 6.5, 'y': -4.5}

LOW_BATTERY_THRESHOLD = 25.0     # send to charger below this %
CRITICAL_BATTERY = 10.0          # refuse new tasks


class TaskAllocator(Node):

    def __init__(self):
        super().__init__('task_allocator')

        self.declare_parameter('num_robots', 3)
        self.declare_parameter('task_interval', 15.0)

        self.num_robots = self.get_parameter('num_robots').value
        self.task_interval = self.get_parameter('task_interval').value

        self.robot_names = [f'robot{i+1}' for i in range(self.num_robots)]

        # State tracking
        self.robot_poses = {}       # name -> (x, y)
        self.robot_battery = {}     # name -> float
        self.robot_busy = {}        # name -> bool
        self.task_queue = []        # pending tasks
        self.completed_tasks = 0
        self.total_energy_used = 0.0

        # Subscribers
        for name in self.robot_names:
            self.create_subscription(
                Odometry, f'/{name}/odom',
                lambda msg, n=name: self._odom_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{name}/battery_level',
                lambda msg, n=name: self._battery_cb(msg, n), 10)

        # Publishers
        self.task_pub = self.create_publisher(String, '/task_assignments', 10)
        self.status_pub = self.create_publisher(String, '/task_allocator/status', 10)

        # Generate tasks periodically
        self.task_timer = self.create_timer(self.task_interval, self._generate_and_assign)
        # Status publishing
        self.status_timer = self.create_timer(5.0, self._publish_status)

        # Init default battery
        for name in self.robot_names:
            self.robot_battery[name] = 100.0
            self.robot_busy[name] = False

        self.get_logger().info(
            f'TaskAllocator started — {self.num_robots} robots, '
            f'interval={self.task_interval}s')

    # ---- callbacks --------------------------------------------------------

    def _odom_cb(self, msg, robot_name):
        self.robot_poses[robot_name] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def _battery_cb(self, msg, robot_name):
        self.robot_battery[robot_name] = msg.data

    # ---- task generation --------------------------------------------------

    def _generate_and_assign(self):
        """Generate a batch of pickup tasks and assign to available robots."""

        # Check which robots need charging
        charging_assignments = []
        for name in self.robot_names:
            bat = self.robot_battery.get(name, 100.0)
            if bat < LOW_BATTERY_THRESHOLD:
                charging_assignments.append({
                    'robot': name,
                    'goal_x': CHARGING_STATION['x'],
                    'goal_y': CHARGING_STATION['y'],
                    'task_type': 'charge',
                })
                self.robot_busy[name] = True
                self.get_logger().warn(
                    f'{name} battery={bat:.1f}% → sending to charger')

        # Available robots
        available = [
            n for n in self.robot_names
            if not self.robot_busy.get(n, False)
            and self.robot_battery.get(n, 100.0) >= CRITICAL_BATTERY
            and n in self.robot_poses
        ]

        if not available:
            if charging_assignments:
                self._publish_assignments(charging_assignments)
            self.get_logger().info('No available robots for new tasks')
            return

        # Generate random pickup tasks (simulate incoming orders)
        num_tasks = min(len(available), random.randint(1, 3))
        shelves = random.sample(SHELF_LOCATIONS, num_tasks)

        # Assign tasks using energy-distance heuristic
        assignments = list(charging_assignments)
        used_robots = set(a['robot'] for a in charging_assignments)

        for shelf in shelves:
            best_robot = None
            best_score = float('inf')

            for name in available:
                if name in used_robots:
                    continue

                rx, ry = self.robot_poses.get(name, (0, 0))
                dist = math.hypot(shelf['x'] - rx, shelf['y'] - ry)
                bat = self.robot_battery.get(name, 100.0)

                # Score = distance / (battery remaining)
                #  -> prefers closer robots with more battery
                score = dist / max(bat, 1.0) * 100.0
                if score < best_score:
                    best_score = score
                    best_robot = name

            if best_robot:
                assignments.append({
                    'robot': best_robot,
                    'goal_x': shelf['x'],
                    'goal_y': shelf['y'],
                    'task_type': 'pickup',
                    'shelf': shelf['name'],
                })
                used_robots.add(best_robot)
                self.robot_busy[best_robot] = True
                self.get_logger().info(
                    f'Assigned {best_robot} → shelf {shelf["name"]} '
                    f'(bat={self.robot_battery.get(best_robot, 0):.0f}%)')

        if assignments:
            self._publish_assignments(assignments)

    def _publish_assignments(self, assignments):
        """Publish task assignments for MAPF planner."""
        msg = String()
        msg.data = json.dumps(assignments)
        self.task_pub.publish(msg)
        self.get_logger().info(f'Published {len(assignments)} task assignments')

    def _publish_status(self):
        """Publish system status."""
        status = {
            'timestamp': time.time(),
            'robots': {},
            'completed_tasks': self.completed_tasks,
            'pending_tasks': len(self.task_queue),
        }
        for name in self.robot_names:
            status['robots'][name] = {
                'battery': round(self.robot_battery.get(name, 0), 1),
                'position': list(self.robot_poses.get(name, (0, 0))),
                'busy': self.robot_busy.get(name, False),
            }

        # Reset busy flags for robots that may have finished
        # (simple timeout — in production, use action feedback)
        for name in self.robot_names:
            if self.robot_busy.get(name, False):
                self.robot_busy[name] = False  # allow re-assignment next cycle

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskAllocator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

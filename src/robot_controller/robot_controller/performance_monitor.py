"""
Swarm Performance Monitor - Collects metrics from all robots and the
CBS space-time planner, publishes aggregated statistics for the
energy-aware distributed swarm coordination system.

Monitors:
  - Battery levels per robot (energy awareness)
  - Distance traveled per robot
  - Path efficiency (planned vs actual distance)
  - Energy consumption rate (distance + turns + acceleration)
  - Robot idle/moving time and utilization
  - CBS planner solve time and conflict statistics
  - Collision near-misses (inter-robot distance < threshold)
  - Swarm health and coordination metrics

Part of: Energy-Aware Distributed Swarm Coordination of Autonomous
         Mobile Robots Using Space-Time Multi-Agent Path Planning
"""

import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
import json


class PerformanceMonitor(Node):

    def __init__(self):
        super().__init__('performance_monitor')

        self.declare_parameter('num_robots', 3)
        self.declare_parameter('collision_threshold', 0.5)
        self.declare_parameter('log_interval', 1.0)

        self.num_robots = self.get_parameter('num_robots').value
        self.collision_thresh = self.get_parameter('collision_threshold').value
        log_interval = self.get_parameter('log_interval').value

        # Per-robot state
        self.robot_data = {}
        for i in range(1, self.num_robots + 1):
            ns = f'robot{i}'
            self.robot_data[ns] = {
                'x': 0.0, 'y': 0.0, 'yaw': 0.0,
                'battery': 100.0,
                'distance_traveled': 0.0,
                'prev_x': None, 'prev_y': None,
                'linear_vel': 0.0, 'angular_vel': 0.0,
                'tasks_completed': 0,
                'task_start_time': None,
                'task_times': [],
                'planned_path_length': 0.0,
                'actual_path_length': 0.0,
                'idle_time': 0.0,
                'moving_time': 0.0,
                'is_idle': True,
                'last_update': time.time(),
                'energy_log': [],
                'distance_log': [],
                'velocity_log': [],
                'near_misses': 0,
            }

            # Subscribe to each robot's topics
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, n=ns: self._odom_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{ns}/battery_level',
                lambda msg, n=ns: self._battery_cb(msg, n), 10)
            self.create_subscription(
                Twist, f'/{ns}/cmd_vel',
                lambda msg, n=ns: self._cmd_cb(msg, n), 10)
            self.create_subscription(
                Path, f'/{ns}/planned_path',
                lambda msg, n=ns: self._path_cb(msg, n), 10)

        # Subscribe to planner stats (CBS space-time metrics)
        self.planner_times = []
        self.planner_conflicts = 0
        self.planner_total_turns = 0
        self.planner_total_waits = 0
        self.planner_makespan = 0
        self.create_subscription(
            String, '/mapf_planner/stats', self._planner_stats_cb, 10)

        # Publishers
        self.metrics_pub = self.create_publisher(String, '/performance_metrics_text', 10)
        self.json_pub = self.create_publisher(String, '/metrics_json', 10)

        # Timers
        self.create_timer(log_interval, self._publish_metrics)
        self.create_timer(5.0, self._check_collisions)

        # Global counters
        self.start_time = time.time()
        self.total_near_misses = 0

        self.get_logger().info(
            f'SwarmMonitor started — tracking {self.num_robots} robots '  
            f'(CBS space-time planner metrics enabled)')

    def _odom_cb(self, msg, ns):
        d = self.robot_data[ns]
        d['x'] = msg.pose.pose.position.x
        d['y'] = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        d['yaw'] = math.atan2(siny, cosy)

        # Track distance
        if d['prev_x'] is not None:
            dx = d['x'] - d['prev_x']
            dy = d['y'] - d['prev_y']
            dist = math.hypot(dx, dy)
            if dist < 1.0:  # filter teleport glitches
                d['distance_traveled'] += dist
                d['actual_path_length'] += dist
        d['prev_x'] = d['x']
        d['prev_y'] = d['y']
        d['last_update'] = time.time()

    def _battery_cb(self, msg, ns):
        d = self.robot_data[ns]
        d['battery'] = msg.data
        d['energy_log'].append((time.time() - self.start_time, msg.data))

    def _cmd_cb(self, msg, ns):
        d = self.robot_data[ns]
        d['linear_vel'] = msg.linear.x
        d['angular_vel'] = msg.angular.z

        now = time.time()
        dt = now - d['last_update']
        if dt > 2.0:
            dt = 0.0

        speed = abs(msg.linear.x) + abs(msg.angular.z)
        if speed < 0.01:
            d['idle_time'] += dt
            d['is_idle'] = True
        else:
            d['moving_time'] += dt
            d['is_idle'] = False

        d['velocity_log'].append(
            (time.time() - self.start_time, msg.linear.x, msg.angular.z))

    def _path_cb(self, msg, ns):
        d = self.robot_data[ns]
        length = 0.0
        poses = msg.poses
        for i in range(1, len(poses)):
            dx = poses[i].pose.position.x - poses[i - 1].pose.position.x
            dy = poses[i].pose.position.y - poses[i - 1].pose.position.y
            length += math.hypot(dx, dy)
        d['planned_path_length'] = length
        d['actual_path_length'] = 0.0  # reset for new path
        d['task_start_time'] = time.time()

    def _planner_stats_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.planner_times.append(data.get('planning_time_ms', 0))
            self.planner_total_turns = data.get('total_turns', 0)
            self.planner_total_waits = data.get('total_waits', 0)
            self.planner_makespan = data.get('max_makespan', 0)
        except json.JSONDecodeError:
            pass

    def _check_collisions(self):
        robots = list(self.robot_data.keys())
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                d1 = self.robot_data[robots[i]]
                d2 = self.robot_data[robots[j]]
                dist = math.hypot(d1['x'] - d2['x'], d1['y'] - d2['y'])
                if dist < self.collision_thresh:
                    self.total_near_misses += 1
                    d1['near_misses'] += 1
                    d2['near_misses'] += 1
                    self.get_logger().warn(
                        f'NEAR-MISS: {robots[i]} <-> {robots[j]} dist={dist:.2f}m')

    def _publish_metrics(self):
        elapsed = time.time() - self.start_time

        metrics = {
            'timestamp': round(elapsed, 2),
            'robots': {},
            'system': {
                'total_near_misses': self.total_near_misses,
                'uptime_seconds': round(elapsed, 1),
                'avg_planner_ms': round(
                    sum(self.planner_times) / max(len(self.planner_times), 1), 2),
            }
        }

        total_battery = 0.0
        total_distance = 0.0
        total_idle = 0.0
        total_moving = 0.0

        for ns, d in self.robot_data.items():
            efficiency = 0.0
            if d['actual_path_length'] > 0.1 and d['planned_path_length'] > 0.1:
                efficiency = min(
                    d['planned_path_length'] / d['actual_path_length'] * 100, 100)

            utilization = 0.0
            total_t = d['moving_time'] + d['idle_time']
            if total_t > 0:
                utilization = d['moving_time'] / total_t * 100

            energy_rate = 0.0
            if d['moving_time'] > 0:
                energy_consumed = 100.0 - d['battery']
                energy_rate = energy_consumed / (d['moving_time'] / 60.0)

            robot_metrics = {
                'position': {'x': round(d['x'], 3), 'y': round(d['y'], 3)},
                'battery_pct': round(d['battery'], 2),
                'distance_traveled_m': round(d['distance_traveled'], 3),
                'linear_vel': round(d['linear_vel'], 3),
                'angular_vel': round(d['angular_vel'], 3),
                'path_efficiency_pct': round(efficiency, 1),
                'utilization_pct': round(utilization, 1),
                'energy_rate_pct_per_min': round(energy_rate, 3),
                'idle_time_s': round(d['idle_time'], 1),
                'moving_time_s': round(d['moving_time'], 1),
                'near_misses': d['near_misses'],
            }
            metrics['robots'][ns] = robot_metrics

            total_battery += d['battery']
            total_distance += d['distance_traveled']
            total_idle += d['idle_time']
            total_moving += d['moving_time']

        n = self.num_robots
        metrics['system']['num_robots'] = n
        metrics['system']['avg_battery_pct'] = round(total_battery / n, 2)
        metrics['system']['total_distance_m'] = round(total_distance, 3)
        metrics['system']['fleet_utilization_pct'] = round(
            total_moving / max(total_moving + total_idle, 1) * 100, 1)
        # CBS space-time planner metrics
        metrics['system']['cbs_total_turns'] = self.planner_total_turns
        metrics['system']['cbs_total_waits'] = self.planner_total_waits
        metrics['system']['cbs_makespan'] = self.planner_makespan

        # Publish as JSON
        json_msg = String()
        json_msg.data = json.dumps(metrics)
        self.json_pub.publish(json_msg)

        # Publish readable summary
        summary = String()
        lines = [f'=== FLEET METRICS @ {elapsed:.0f}s ===']
        for ns, rm in metrics['robots'].items():
            lines.append(
                f"  {ns}: bat={rm['battery_pct']:.1f}% | "
                f"dist={rm['distance_traveled_m']:.2f}m | "
                f"vel={rm['linear_vel']:.2f} | "
                f"eff={rm['path_efficiency_pct']:.0f}% | "
                f"util={rm['utilization_pct']:.0f}%"
            )
        lines.append(
            f"  FLEET: avg_bat={metrics['system']['avg_battery_pct']:.1f}% | "
            f"total_dist={metrics['system']['total_distance_m']:.2f}m | "
            f"util={metrics['system']['fleet_utilization_pct']:.0f}% | "
            f"near_misses={self.total_near_misses}"
        )
        summary.data = '\n'.join(lines)
        self.metrics_pub.publish(summary)

        self.get_logger().info(lines[0])
        for line in lines[1:]:
            self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitor()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

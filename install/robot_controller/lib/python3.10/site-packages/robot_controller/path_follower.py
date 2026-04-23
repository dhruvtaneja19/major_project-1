"""
Path-following controller for warehouse robots.

Subscribes to:
  /<robot_ns>/planned_path  (nav_msgs/Path)
  /<robot_ns>/odom          (nav_msgs/Odometry)

Publishes:
  /<robot_ns>/cmd_vel       (geometry_msgs/Twist)
  /<robot_ns>/battery_level (std_msgs/Float32)  — simulated battery

Uses pure-pursuit style control with energy tracking.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        self.declare_parameter('robot_namespace', 'robot1')
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('linear_speed', 0.35)
        self.declare_parameter('angular_gain', 2.5)
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('battery_capacity', 100.0)
        self.declare_parameter('energy_per_meter', 0.8)
        self.declare_parameter('idle_drain_rate', 0.01)

        ns = self.get_parameter('robot_namespace').value
        self.lookahead = self.get_parameter('lookahead_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_gain = self.get_parameter('angular_gain').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.battery = self.get_parameter('battery_capacity').value
        self.energy_per_m = self.get_parameter('energy_per_meter').value
        self.idle_drain = self.get_parameter('idle_drain_rate').value

        self.ns = ns

        # State
        self.path = []
        self.path_index = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.has_odom = False
        self.active = False
        self.prev_x = None
        self.prev_y = None

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, f'/{ns}/cmd_vel', 10)
        self.battery_pub = self.create_publisher(Float32, f'/{ns}/battery_level', 10)

        # Subscribers
        self.create_subscription(Path, f'/{ns}/planned_path', self._path_cb, 10)
        self.create_subscription(Odometry, f'/{ns}/odom', self._odom_cb, 10)

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self._control_loop)
        # Battery publish at 1 Hz
        self.bat_timer = self.create_timer(1.0, self._publish_battery)

        self.get_logger().info(f'PathFollower [{ns}] started — battery={self.battery:.1f}%')

    # ---- callbacks --------------------------------------------------------

    def _path_cb(self, msg):
        self.path = []
        for ps in msg.poses:
            self.path.append((ps.pose.position.x, ps.pose.position.y))
        self.path_index = 0
        self.active = len(self.path) > 0
        self.get_logger().info(f'Received path with {len(self.path)} waypoints')

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_odom = True

    # ---- control ----------------------------------------------------------

    def _control_loop(self):
        if not self.has_odom or not self.active or not self.path:
            return

        # Energy tracking
        if self.prev_x is not None:
            dist = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
            self.battery -= dist * self.energy_per_m
        self.battery -= self.idle_drain
        self.battery = max(self.battery, 0.0)
        self.prev_x = self.x
        self.prev_y = self.y

        # Low battery — stop
        if self.battery < 5.0:
            self._stop()
            self.active = False
            self.get_logger().warn(f'[{self.ns}] LOW BATTERY ({self.battery:.1f}%) — stopping')
            return

        # Find lookahead point
        target = self._get_lookahead_point()
        if target is None:
            self._stop()
            self.active = False
            self.get_logger().info(f'[{self.ns}] Reached goal! battery={self.battery:.1f}%')
            return

        # Pure pursuit
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.hypot(dx, dy)

        # Desired heading
        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.yaw

        # Normalize to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        cmd = Twist()

        if abs(yaw_error) > 0.4:
            # Turn in place
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_gain * yaw_error
        else:
            # Drive forward with proportional steering
            cmd.linear.x = min(self.linear_speed, dist)
            cmd.angular.z = self.angular_gain * yaw_error

        # Clamp angular velocity
        cmd.angular.z = max(-1.5, min(1.5, cmd.angular.z))

        self.cmd_pub.publish(cmd)

    def _get_lookahead_point(self):
        """Find the first path point beyond lookahead distance, advance index."""
        while self.path_index < len(self.path):
            px, py = self.path[self.path_index]
            dist = math.hypot(px - self.x, py - self.y)

            if dist < self.goal_tolerance and self.path_index == len(self.path) - 1:
                return None  # reached final goal

            if dist >= self.lookahead:
                return (px, py)

            self.path_index += 1

        # Return last point
        if self.path:
            px, py = self.path[-1]
            dist = math.hypot(px - self.x, py - self.y)
            if dist < self.goal_tolerance:
                return None
            return (px, py)
        return None

    def _stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _publish_battery(self):
        msg = Float32()
        msg.data = self.battery
        self.battery_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
Live Swarm Dashboard — Subscribes to /metrics_json and generates
real-time matplotlib plots for the energy-aware distributed swarm
coordination system.

Plots (8 panels):
  1. Battery Level vs Time (all robots)
  2. Distance Traveled vs Time (all robots)
  3. Robot Velocities over Time
  4. Fleet Utilization vs Time
  5. Energy Consumption Rate vs Time
  6. Path Efficiency Comparison (bar chart)
  7. Robot Trajectories (map view)
  8. Swarm KPI Summary (CBS + energy metrics)

Dynamically adapts to 3-5 robots based on incoming metrics.

Run:  ros2 run robot_controller live_plotter
"""

import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np


# Colors for up to 5 robots
ROBOT_COLORS = {
    'robot1': '#2196F3',  # blue
    'robot2': '#FF5722',  # red-orange
    'robot3': '#4CAF50',  # green
    'robot4': '#9C27B0',  # purple
    'robot5': '#FF9800',  # amber
}


class LivePlotter(Node):

    def __init__(self):
        super().__init__('live_plotter')

        # Data buffers — dynamically populated from metrics
        self.timestamps = []
        self.battery_data = {}
        self.distance_data = {}
        self.velocity_data = {}
        self.utilization_data = []
        self.fleet_battery_data = []
        self.energy_rate_data = {}
        self.path_efficiency = {}
        self.positions = {}
        self.near_misses = 0
        self.total_distance = []
        self.avg_planner_ms = 0.0
        self.num_robots = 0
        self.cbs_turns = 0
        self.cbs_waits = 0
        self.cbs_makespan = 0
        self.robot_names = []

        self.lock = threading.Lock()

        self.create_subscription(String, '/metrics_json', self._metrics_cb, 10)

        self.get_logger().info(
            'Swarm LivePlotter started — waiting for metrics...')

    def _metrics_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        with self.lock:
            t = data['timestamp']
            self.timestamps.append(t)

            # Dynamically discover robots from metrics
            for ns in sorted(data.get('robots', {}).keys()):
                if ns not in self.robot_names:
                    self.robot_names.append(ns)
                    self.battery_data[ns] = []
                    self.distance_data[ns] = []
                    self.velocity_data[ns] = []
                    self.energy_rate_data[ns] = []
                    self.path_efficiency[ns] = 0.0
                    self.positions[ns] = []

                r = data['robots'][ns]
                self.battery_data[ns].append(r['battery_pct'])
                self.distance_data[ns].append(r['distance_traveled_m'])
                self.velocity_data[ns].append(r['linear_vel'])
                self.energy_rate_data[ns].append(r['energy_rate_pct_per_min'])
                self.path_efficiency[ns] = r['path_efficiency_pct']
                self.positions[ns].append(
                    (r['position']['x'], r['position']['y']))

            sys_data = data['system']
            self.utilization_data.append(sys_data['fleet_utilization_pct'])
            self.fleet_battery_data.append(sys_data['avg_battery_pct'])
            self.total_distance.append(sys_data['total_distance_m'])
            self.near_misses = sys_data['total_near_misses']
            self.avg_planner_ms = sys_data.get('avg_planner_ms', 0)
            self.num_robots = sys_data.get('num_robots', len(self.robot_names))
            self.cbs_turns = sys_data.get('cbs_total_turns', 0)
            self.cbs_waits = sys_data.get('cbs_total_waits', 0)
            self.cbs_makespan = sys_data.get('cbs_makespan', 0)

    def _get_color(self, ns):
        return ROBOT_COLORS.get(ns, '#FFFFFF')

    def run_plot(self):
        """Set up matplotlib figure and start animation."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.suptitle(
            'Energy-Aware Swarm Coordination — Space-Time MAPF Dashboard',
            fontsize=16, fontweight='bold', color='#FFD600')

        gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

        self.ax_bat = self.fig.add_subplot(gs[0, 0])
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_vel = self.fig.add_subplot(gs[0, 2])
        self.ax_util = self.fig.add_subplot(gs[1, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 1])
        self.ax_eff = self.fig.add_subplot(gs[1, 2])
        self.ax_map = self.fig.add_subplot(gs[2, 0])
        self.ax_kpi = self.fig.add_subplot(gs[2, 1:])

        self.anim = FuncAnimation(
            self.fig, self._update_plots, interval=1000, cache_frame_data=False)
        plt.show()

    def _update_plots(self, frame):
        with self.lock:
            if len(self.timestamps) < 2:
                return

            t = list(self.timestamps)

            # --- 1: Battery Level ---
            self.ax_bat.clear()
            for ns in self.robot_names:
                d = self.battery_data.get(ns, [])
                if d:
                    self.ax_bat.plot(t[:len(d)], d, color=self._get_color(ns),
                                    linewidth=2, label=ns)
            self.ax_bat.axhline(y=25, color='red', linestyle='--',
                                alpha=0.5, label='Low threshold')
            self.ax_bat.set_title('Battery Level', fontsize=11, fontweight='bold')
            self.ax_bat.set_xlabel('Time (s)')
            self.ax_bat.set_ylabel('Battery (%)')
            self.ax_bat.set_ylim(0, 105)
            self.ax_bat.legend(loc='lower left', fontsize=7)
            self.ax_bat.grid(True, alpha=0.3)

            # --- 2: Distance Traveled ---
            self.ax_dist.clear()
            for ns in self.robot_names:
                d = self.distance_data.get(ns, [])
                if d:
                    self.ax_dist.plot(t[:len(d)], d, color=self._get_color(ns),
                                     linewidth=2, label=ns)
            self.ax_dist.set_title('Distance Traveled', fontsize=11,
                                   fontweight='bold')
            self.ax_dist.set_xlabel('Time (s)')
            self.ax_dist.set_ylabel('Distance (m)')
            self.ax_dist.legend(loc='upper left', fontsize=7)
            self.ax_dist.grid(True, alpha=0.3)

            # --- 3: Robot Velocities ---
            self.ax_vel.clear()
            for ns in self.robot_names:
                d = self.velocity_data.get(ns, [])
                if d:
                    self.ax_vel.plot(t[:len(d)], d, color=self._get_color(ns),
                                    linewidth=1.5, label=ns, alpha=0.8)
            self.ax_vel.set_title('Linear Velocity', fontsize=11,
                                  fontweight='bold')
            self.ax_vel.set_xlabel('Time (s)')
            self.ax_vel.set_ylabel('Velocity (m/s)')
            self.ax_vel.legend(loc='upper right', fontsize=7)
            self.ax_vel.grid(True, alpha=0.3)

            # --- 4: Fleet Utilization ---
            self.ax_util.clear()
            if self.utilization_data:
                ut = self.utilization_data
                self.ax_util.fill_between(t[:len(ut)], ut,
                                          alpha=0.3, color='#FFD600')
                self.ax_util.plot(t[:len(ut)], ut,
                                  color='#FFD600', linewidth=2)
            self.ax_util.set_title('Fleet Utilization', fontsize=11,
                                   fontweight='bold')
            self.ax_util.set_xlabel('Time (s)')
            self.ax_util.set_ylabel('Utilization (%)')
            self.ax_util.set_ylim(0, 105)
            self.ax_util.grid(True, alpha=0.3)

            # --- 5: Energy Consumption Rate ---
            self.ax_energy.clear()
            for ns in self.robot_names:
                d = self.energy_rate_data.get(ns, [])
                if d:
                    self.ax_energy.plot(t[:len(d)], d, color=self._get_color(ns),
                                       linewidth=2, label=ns)
            self.ax_energy.set_title('Energy Drain Rate', fontsize=11,
                                     fontweight='bold')
            self.ax_energy.set_xlabel('Time (s)')
            self.ax_energy.set_ylabel('% / min')
            self.ax_energy.legend(loc='upper right', fontsize=7)
            self.ax_energy.grid(True, alpha=0.3)

            # --- 6: Path Efficiency (Bar Chart) ---
            self.ax_eff.clear()
            robots = list(self.path_efficiency.keys())
            effs = [self.path_efficiency[r] for r in robots]
            colors = [self._get_color(r) for r in robots]
            if robots:
                bars = self.ax_eff.bar(
                    robots, effs, color=colors,
                    edgecolor='white', linewidth=0.5)
                for bar, val in zip(bars, effs):
                    self.ax_eff.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
            self.ax_eff.set_title('Path Efficiency', fontsize=11,
                                  fontweight='bold')
            self.ax_eff.set_ylabel('Efficiency (%)')
            self.ax_eff.set_ylim(0, 110)
            self.ax_eff.grid(True, alpha=0.3, axis='y')

            # --- 7: Robot Trajectories ---
            self.ax_map.clear()
            # Draw pillar obstacles
            pillar_positions = [
                (-4, 3), (0, 3), (4, 3),
                (-4, -3), (0, -3), (4, -3),
            ]
            for px, py in pillar_positions:
                rect = plt.Rectangle(
                    (px - 0.75, py - 0.75), 1.5, 1.5,
                    fill=True, facecolor='#555555',
                    edgecolor='#888888', linewidth=0.5)
                self.ax_map.add_patch(rect)

            # Central barrier
            self.ax_map.add_patch(plt.Rectangle(
                (-2.0, -0.15), 4.0, 0.3,
                fill=True, facecolor='#887722',
                edgecolor='#AAAA44', linewidth=0.5))

            # Arena walls
            self.ax_map.add_patch(plt.Rectangle(
                (-8, -6), 16, 12, fill=False,
                edgecolor='white', linewidth=1.5))

            # Zone markers
            zones = [
                (-6, 4.5, 'A', '#66BB6A'),
                (6, 4.5, 'B', '#42A5F5'),
                (6, -4.5, 'C', '#EF5350'),
                (-6, -4.5, 'D', '#FFD600'),
            ]
            for zx, zy, label, color in zones:
                self.ax_map.scatter(zx, zy, color=color, s=60,
                                    marker='s', zorder=3, alpha=0.7)
                self.ax_map.text(zx, zy + 0.5, label, fontsize=7,
                                 ha='center', color=color, fontweight='bold')

            for ns in self.robot_names:
                color = self._get_color(ns)
                pos = self.positions.get(ns, [])
                if len(pos) > 1:
                    xs = [p[0] for p in pos]
                    ys = [p[1] for p in pos]
                    self.ax_map.plot(xs, ys, color=color,
                                    alpha=0.4, linewidth=1)
                    self.ax_map.scatter(
                        xs[-1], ys[-1], color=color, s=80,
                        zorder=5, edgecolors='white',
                        linewidth=1.5, label=ns)

            self.ax_map.set_title('Robot Trajectories', fontsize=11,
                                  fontweight='bold')
            self.ax_map.set_xlabel('X (m)')
            self.ax_map.set_ylabel('Y (m)')
            self.ax_map.set_xlim(-9, 9)
            self.ax_map.set_ylim(-7, 7)
            self.ax_map.set_aspect('equal')
            self.ax_map.legend(loc='upper right', fontsize=7)
            self.ax_map.grid(True, alpha=0.2)

            # --- 8: KPI Summary ---
            self.ax_kpi.clear()
            self.ax_kpi.axis('off')

            latest_util = (self.utilization_data[-1]
                           if self.utilization_data else 0)
            latest_fleet_bat = (self.fleet_battery_data[-1]
                                if self.fleet_battery_data else 100)
            latest_total_dist = (self.total_distance[-1]
                                 if self.total_distance else 0)
            elapsed = t[-1] if t else 0
            avg_eff = float(np.mean(
                [v for v in self.path_efficiency.values()])) if any(
                    self.path_efficiency.values()) else 0
            throughput = latest_total_dist / max(elapsed, 1) * 60
            n_robots = self.num_robots or len(self.robot_names)

            kpi_text = (
                f"{'=' * 50}\n"
                f"  SWARM KPIs @ {elapsed:.0f}s\n"
                f"{'=' * 50}\n\n"
                f"  Active Robots:         {n_robots:5d}\n"
                f"  Fleet Avg Battery:     {latest_fleet_bat:6.1f} %\n"
                f"  Total Distance:        {latest_total_dist:6.2f} m\n"
                f"  Fleet Utilization:     {latest_util:6.1f} %\n"
                f"  Avg Path Efficiency:   {avg_eff:6.1f} %\n"
                f"  Throughput:            {throughput:6.2f} m/min\n"
                f"  Near-Miss Events:      {self.near_misses:5d}\n\n"
                f"{'=' * 50}\n"
                f"  CBS SPACE-TIME PLANNER\n"
                f"{'=' * 50}\n"
                f"  Avg Plan Time:         {self.avg_planner_ms:6.2f} ms\n"
                f"  Makespan:              {self.cbs_makespan:5d} steps\n"
                f"  Path Turns:            {self.cbs_turns:5d}\n"
                f"  Wait Actions:          {self.cbs_waits:5d}\n\n"
                f"{'=' * 50}\n"
                f"  OPTIMIZATION TARGETS\n"
                f"{'=' * 50}\n"
                f"  Utilization > 70%:  "
                f"{'PASS' if latest_util > 70 else 'FAIL'}\n"
                f"  Battery > 20%:      "
                f"{'PASS' if latest_fleet_bat > 20 else 'FAIL'}\n"
                f"  Path Eff > 80%:     "
                f"{'PASS' if avg_eff > 80 else 'FAIL'}\n"
                f"  Zero Collisions:    "
                f"{'PASS' if self.near_misses == 0 else 'FAIL'}\n"
            )

            self.ax_kpi.text(
                0.05, 0.95, kpi_text,
                transform=self.ax_kpi.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='top', color='#E0E0E0',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#1a1a2e', alpha=0.9))


def main(args=None):
    rclpy.init(args=args)
    node = LivePlotter()

    # Spin ROS in a background thread so matplotlib can use the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Run matplotlib in main thread (required by most backends)
    node.run_plot()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

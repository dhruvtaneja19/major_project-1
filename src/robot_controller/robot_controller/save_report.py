"""
Swarm Performance Report Generator — Collects metrics for a configurable
duration then saves a full PDF report with optimization charts + CSV data.

Supports dynamic 3-5 robot swarms. Includes CBS space-time planner
statistics and energy-aware cost analysis.

Usage:
  ros2 run robot_controller save_report
  ros2 run robot_controller save_report --ros-args -p duration_seconds:=60

Output goes to ~/mapf_ws/reports/
"""

import json
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime


# Colors for up to 5 robots
ROBOT_COLORS = {
    'robot1': '#2196F3', 'robot2': '#FF5722', 'robot3': '#4CAF50',
    'robot4': '#9C27B0', 'robot5': '#FF9800',
}


class ReportGenerator(Node):

    def __init__(self):
        super().__init__('report_generator')

        self.declare_parameter('duration_seconds', 120)
        self.duration = self.get_parameter('duration_seconds').value

        # Dynamic data buffers — populated from metrics
        self.timestamps = []
        self.battery = {}
        self.distance = {}
        self.velocity = {}
        self.energy_rate = {}
        self.utilization = []
        self.fleet_battery = []
        self.positions = {}
        self.path_eff = {}
        self.near_misses = 0
        self.total_distance_list = []
        self.avg_planner_ms = 0.0
        self.cbs_turns = 0
        self.cbs_waits = 0
        self.cbs_makespan = 0
        self.num_robots = 0
        self.robot_names = []
        self.done = False

        self.create_subscription(String, '/metrics_json', self._cb, 10)
        self.start_time = time.time()
        self.create_timer(2.0, self._check_done)

        self.get_logger().info(
            f'ReportGenerator: collecting data for {self.duration}s '
            f'then saving PDF + CSV...')

    def _cb(self, msg):
        if self.done:
            return
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.timestamps.append(d['timestamp'])

        # Dynamically discover robots
        for ns in sorted(d.get('robots', {}).keys()):
            if ns not in self.robot_names:
                self.robot_names.append(ns)
                self.battery[ns] = []
                self.distance[ns] = []
                self.velocity[ns] = []
                self.energy_rate[ns] = []
                self.path_eff[ns] = 0.0
                self.positions[ns] = []

            r = d['robots'][ns]
            self.battery[ns].append(r['battery_pct'])
            self.distance[ns].append(r['distance_traveled_m'])
            self.velocity[ns].append(r['linear_vel'])
            self.energy_rate[ns].append(r['energy_rate_pct_per_min'])
            self.path_eff[ns] = r['path_efficiency_pct']
            self.positions[ns].append(
                (r['position']['x'], r['position']['y']))

        sys_d = d['system']
        self.utilization.append(sys_d['fleet_utilization_pct'])
        self.fleet_battery.append(sys_d['avg_battery_pct'])
        self.total_distance_list.append(sys_d['total_distance_m'])
        self.near_misses = sys_d['total_near_misses']
        self.avg_planner_ms = sys_d.get('avg_planner_ms', 0.0)
        self.num_robots = sys_d.get('num_robots', len(self.robot_names))
        self.cbs_turns = sys_d.get('cbs_total_turns', 0)
        self.cbs_waits = sys_d.get('cbs_total_waits', 0)
        self.cbs_makespan = sys_d.get('cbs_makespan', 0)

    def _check_done(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration and len(self.timestamps) > 5 and not self.done:
            self.done = True
            self.get_logger().info(
                f'Collected {len(self.timestamps)} samples over '
                f'{elapsed:.0f}s — generating report...')
            self._generate_report()

    def _generate_report(self):
        report_dir = os.path.expanduser('~/mapf_ws/reports')
        os.makedirs(report_dir, exist_ok=True)
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        robots = self.robot_names

        def _color(ns):
            return ROBOT_COLORS.get(ns, '#FFFFFF')

        # ---- Save CSV ----
        csv_path = os.path.join(report_dir, f'metrics_{ts_str}.csv')
        with open(csv_path, 'w') as f:
            f.write(
                'time,robot,battery,distance,velocity,'
                'energy_rate,utilization\n')
            for i, t in enumerate(self.timestamps):
                for ns in robots:
                    if i < len(self.battery.get(ns, [])):
                        util = (self.utilization[i]
                                if i < len(self.utilization) else 0)
                        f.write(
                            f"{t:.1f},{ns},{self.battery[ns][i]:.2f},"
                            f"{self.distance[ns][i]:.3f},"
                            f"{self.velocity[ns][i]:.3f},"
                            f"{self.energy_rate[ns][i]:.4f},"
                            f"{util:.1f}\n")
        self.get_logger().info(f'CSV saved: {csv_path}')

        # ---- Generate PDF ----
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 20))
        fig.suptitle(
            'Energy-Aware Swarm Coordination — Performance Report\n'
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  '
            f'Duration: {self.timestamps[-1]:.0f}s  |  '
            f'Robots: {len(robots)}',
            fontsize=14, fontweight='bold', color='#FFD600', y=0.98)

        gs = gridspec.GridSpec(4, 2, hspace=0.38, wspace=0.3)
        t = self.timestamps

        # 1. Battery
        ax = fig.add_subplot(gs[0, 0])
        for ns in robots:
            d = self.battery.get(ns, [])
            if d:
                ax.plot(t[:len(d)], d, color=_color(ns), lw=2, label=ns)
        ax.axhline(25, color='red', ls='--', alpha=0.5, label='Low')
        ax.set_title('Battery Drain Over Time', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Battery (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Distance
        ax = fig.add_subplot(gs[0, 1])
        for ns in robots:
            d = self.distance.get(ns, [])
            if d:
                ax.plot(t[:len(d)], d, color=_color(ns), lw=2, label=ns)
        ax.set_title('Cumulative Distance Traveled', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Velocity
        ax = fig.add_subplot(gs[1, 0])
        for ns in robots:
            d = self.velocity.get(ns, [])
            if d:
                ax.plot(t[:len(d)], d, color=_color(ns), lw=1.5,
                        alpha=0.8, label=ns)
        ax.set_title('Linear Velocity Profile', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Utilization
        ax = fig.add_subplot(gs[1, 1])
        if self.utilization:
            ax.fill_between(t[:len(self.utilization)], self.utilization,
                            alpha=0.3, color='#FFD600')
            ax.plot(t[:len(self.utilization)], self.utilization,
                    color='#FFD600', lw=2)
        ax.set_title('Fleet Utilization', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 5. Energy Rate
        ax = fig.add_subplot(gs[2, 0])
        for ns in robots:
            d = self.energy_rate.get(ns, [])
            if d:
                ax.plot(t[:len(d)], d, color=_color(ns), lw=2, label=ns)
        ax.set_title('Energy Consumption Rate', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('% per minute')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6. Path Efficiency Bar
        ax = fig.add_subplot(gs[2, 1])
        vals = [self.path_eff.get(r, 0) for r in robots]
        bars = ax.bar(robots, vals,
                      color=[_color(r) for r in robots],
                      edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{v:.0f}%', ha='center', fontweight='bold')
        ax.set_title('Path Efficiency', fontweight='bold')
        ax.set_ylabel('Efficiency (%)')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        # 7. Trajectory map
        ax = fig.add_subplot(gs[3, 0])
        pillar_positions = [
            (-4, 3), (0, 3), (4, 3),
            (-4, -3), (0, -3), (4, -3),
        ]
        for px, py in pillar_positions:
            ax.add_patch(plt.Rectangle(
                (px - 0.75, py - 0.75), 1.5, 1.5,
                fill=True, facecolor='#555555',
                edgecolor='#888888', linewidth=0.5))
        # Central barrier
        ax.add_patch(plt.Rectangle(
            (-2.0, -0.15), 4.0, 0.3,
            fill=True, facecolor='#887722',
            edgecolor='#AAAA44', linewidth=0.5))
        # Arena walls
        ax.add_patch(plt.Rectangle(
            (-8, -6), 16, 12, fill=False,
            edgecolor='white', linewidth=1.5))

        for ns in robots:
            c = _color(ns)
            pos = self.positions.get(ns, [])
            if pos:
                xs = [p[0] for p in pos]
                ys = [p[1] for p in pos]
                ax.plot(xs, ys, color=c, alpha=0.4, lw=1)
                ax.scatter(xs[-1], ys[-1], color=c, s=80, zorder=5,
                           edgecolors='white', lw=1.5, label=ns)
                ax.scatter(xs[0], ys[0], color=c, s=40, marker='s',
                           zorder=4, alpha=0.6)
        ax.set_title('Robot Trajectories', fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(-9, 9)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # 8. KPI Summary
        ax = fig.add_subplot(gs[3, 1])
        ax.axis('off')
        avg_bat = float(np.mean(self.fleet_battery)) if self.fleet_battery else 100
        final_bat = self.fleet_battery[-1] if self.fleet_battery else 100
        max_dist = max(
            [self.distance[ns][-1] if self.distance.get(ns) else 0
             for ns in robots], default=0)
        avg_util = float(np.mean(self.utilization)) if self.utilization else 0
        avg_eff = float(np.mean(list(self.path_eff.values()))) if self.path_eff else 0
        elapsed = self.timestamps[-1] if self.timestamps else 0
        throughput = max_dist / max(elapsed, 1) * 60

        stars = lambda v, mx=5: '*' * min(int(v / 20), mx)

        kpi = (
            f"{'=' * 48}\n"
            f"  FINAL SWARM KPIs\n"
            f"{'=' * 48}\n\n"
            f"  Swarm Size:            {len(robots):>8d}\n"
            f"  Duration:              {elapsed:>8.0f} s\n"
            f"  Avg Fleet Battery:     {avg_bat:>8.1f} %\n"
            f"  Final Fleet Battery:   {final_bat:>8.1f} %\n"
            f"  Max Robot Distance:    {max_dist:>8.2f} m\n"
            f"  Avg Utilization:       {avg_util:>8.1f} %\n"
            f"  Avg Path Efficiency:   {avg_eff:>8.1f} %\n"
            f"  Throughput:            {throughput:>8.2f} m/min\n"
            f"  Near-Miss Events:      {self.near_misses:>8d}\n\n"
            f"{'=' * 48}\n"
            f"  CBS SPACE-TIME PLANNER\n"
            f"{'=' * 48}\n"
            f"  Avg Plan Time:         {self.avg_planner_ms:>8.2f} ms\n"
            f"  Makespan:              {self.cbs_makespan:>8d} steps\n"
            f"  Path Turns:            {self.cbs_turns:>8d}\n"
            f"  Wait Actions:          {self.cbs_waits:>8d}\n\n"
            f"{'=' * 48}\n"
            f"  OPTIMIZATION SCORES\n"
            f"{'=' * 48}\n"
            f"  Energy Efficiency:  {stars(avg_bat)}\n"
            f"  Utilization:        {stars(avg_util)}\n"
            f"  Path Quality:       {stars(avg_eff)}\n"
            f"  Safety:             "
            f"{'*****' if self.near_misses == 0 else stars(max(100 - self.near_misses * 20, 20))}\n"
        )
        ax.text(0.05, 0.95, kpi, transform=ax.transAxes,
                fontsize=9, fontfamily='monospace',
                va='top', color='#E0E0E0',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#1a1a2e', alpha=0.9))

        pdf_path = os.path.join(report_dir, f'report_{ts_str}.pdf')
        fig.savefig(pdf_path, dpi=150, bbox_inches='tight',
                    facecolor='#121212')

        png_path = os.path.join(report_dir, f'report_{ts_str}.png')
        fig.savefig(png_path, dpi=120, bbox_inches='tight',
                    facecolor='#121212')
        plt.close(fig)

        self.get_logger().info(f'PDF report saved: {pdf_path}')
        self.get_logger().info(f'PNG report saved: {png_path}')
        self.get_logger().info(f'CSV data saved:   {csv_path}')
        self.get_logger().info('=== Report generation complete! ===')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = ReportGenerator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

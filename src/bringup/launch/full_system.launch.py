"""
Full System Bringup for:
  Energy-Aware Distributed Swarm Coordination of Autonomous Mobile Robots
  Using Space-Time Multi-Agent Path Planning

Launches:
  1. Gazebo simulation (MAPF arena world + N robots)
  2. CBS Space-Time MAPF Planner with energy-aware cost
  3. Distributed path follower controllers (one per robot)
  4. Energy-aware swarm task allocator
  5. Performance monitor for fleet metrics
  6. (Optional) Live plotter for real-time dashboard

Supports 3-5 robots via NUM_ROBOTS parameter.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


# Default swarm size (3-5 robots)
DEFAULT_NUM_ROBOTS = 3


def generate_launch_description():

    # ── Launch arguments ──
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', default_value=str(DEFAULT_NUM_ROBOTS),
        description='Number of robots in the swarm (3-5)')

    # We need the integer for loop generation — use default for static launch
    NUM_ROBOTS = DEFAULT_NUM_ROBOTS

    # --- 1. Gazebo simulation (warehouse world + robot spawning) ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('warehouse_gazebo'),
                'launch',
                'sim.launch.py'
            )
        )
    )

    # --- 2. CBS Space-Time MAPF Planner ---
    mapf_planner_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='mapf_planner',
                executable='mapf_planner',
                name='mapf_planner',
                parameters=[{
                    'num_robots': NUM_ROBOTS,
                    'energy_weight_distance': 1.0,
                    'energy_weight_turns': 0.3,
                    'energy_weight_wait': 0.2,
                    'energy_weight_congestion': 0.4,
                    'max_planning_time_steps': 120,
                    'max_cbs_nodes': 5000,
                }],
                output='screen',
            )
        ]
    )

    # --- 3. Distributed path followers (one per robot) ---
    path_followers = []
    for i in range(1, NUM_ROBOTS + 1):
        pf = TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='robot_controller',
                    executable='path_follower',
                    name=f'path_follower_robot{i}',
                    parameters=[{
                        'robot_namespace': f'robot{i}',
                        'linear_speed': 0.35,
                        'lookahead_distance': 0.4,
                        'goal_tolerance': 0.25,
                        'battery_capacity': 100.0,
                        'energy_per_meter': 0.8,
                        'energy_per_turn': 0.3,
                        'idle_drain_rate': 0.01,
                        'acceleration_energy': 0.15,
                    }],
                    output='screen',
                )
            ]
        )
        path_followers.append(pf)

    # --- 4. Swarm Task Allocator ---
    task_allocator_node = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='task_allocator',
                executable='task_allocator',
                name='task_allocator',
                parameters=[{
                    'num_robots': NUM_ROBOTS,
                    'task_interval': 15.0,
                    'energy_weight': 0.4,
                    'distance_weight': 0.4,
                    'congestion_weight': 0.2,
                }],
                output='screen',
            )
        ]
    )

    # --- 5. Performance Monitor ---
    performance_monitor_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='robot_controller',
                executable='performance_monitor',
                name='performance_monitor',
                parameters=[{
                    'num_robots': NUM_ROBOTS,
                    'collision_threshold': 0.5,
                }],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        num_robots_arg,
        gazebo_launch,
        mapf_planner_node,
        *path_followers,
        task_allocator_node,
        performance_monitor_node,
    ])

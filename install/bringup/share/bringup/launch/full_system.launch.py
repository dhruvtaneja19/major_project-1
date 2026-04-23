"""
Full system bringup for Adaptive Multi-Robot Navigation and
Energy-Efficient Coordination in Smart Warehouse Systems.

Launches:
  1. Gazebo simulation (warehouse world + 3 robots)
  2. CBS MAPF Planner node
  3. Path follower controller for each robot
  4. Energy-aware task allocator
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


NUM_ROBOTS = 3


def generate_launch_description():
    planner_params = os.path.join(
        get_package_share_directory('mapf_planner'),
        'config',
        'aco_params.yaml'
    )

    # --- 1. Gazebo simulation (includes robot spawning) ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('warehouse_gazebo'),
                'launch',
                'sim.launch.py'
            )
        )
    )

    # --- 2. MAPF Planner (CBS) — start after robots are spawned ---
    mapf_planner_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='mapf_planner',
                executable='mapf_planner',
                name='mapf_planner',
                parameters=[planner_params, {'num_robots': NUM_ROBOTS}],
                output='screen',
            )
        ]
    )

    # --- 3. Path followers for each robot ---
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
                    }],
                    output='screen',
                )
            ]
        )
        path_followers.append(pf)

    # --- 4. Task Allocator — start after everything else ---
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
                }],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        gazebo_launch,
        mapf_planner_node,
        *path_followers,
        task_allocator_node,
    ])

"""Launch file for MAPF arena simulation.
Spawns Gazebo world + 3 autonomous mobile robots at different positions
for Space-Time Multi-Agent Path Planning experiments.
"""

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    gazebo_pkg = get_package_share_directory('gazebo_ros')
    desc_pkg = get_package_share_directory('warehouse_description')
    gazebo_world_pkg = get_package_share_directory('warehouse_gazebo')

    world_file = os.path.join(gazebo_world_pkg, 'worlds', 'mapf_arena.world')
    robot_urdf = os.path.join(desc_pkg, 'urdf', 'warehouse_robot.urdf')

    # Read URDF for robot_state_publisher
    with open(robot_urdf, 'r') as f:
        robot_desc = f.read()

    # --- Gazebo server + client ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # --- Robot spawn positions (spread across arena) ---
    robots = [
        {'name': 'robot1', 'x': '-6.0', 'y': '4.5', 'z': '0.1', 'yaw': '0.0'},
        {'name': 'robot2', 'x': '-6.0', 'y': '0.0', 'z': '0.1', 'yaw': '0.0'},
        {'name': 'robot3', 'x': '-6.0', 'y': '-4.5', 'z': '0.1', 'yaw': '0.0'},
    ]

    spawn_nodes = []
    for i, robot in enumerate(robots):
        # Robot state publisher for each robot
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=robot['name'],
            parameters=[{'robot_description': robot_desc,
                          'frame_prefix': robot['name'] + '/'}],
            output='screen',
        )

        # Spawn entity after delay so Gazebo is ready
        spawn = TimerAction(
            period=float(3.0 + i * 2.0),
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', robot['name'],
                        '-file', robot_urdf,
                        '-robot_namespace', robot['name'],
                        '-x', robot['x'],
                        '-y', robot['y'],
                        '-z', robot['z'],
                        '-Y', robot['yaw'],
                    ],
                    output='screen',
                )
            ],
        )

        spawn_nodes.append(rsp)
        spawn_nodes.append(spawn)

    return LaunchDescription([
        gazebo_launch,
    ] + spawn_nodes)

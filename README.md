# Energy-Aware Distributed Swarm Coordination of Autonomous Mobile Robots Using Space-Time Multi-Agent Path Planning

A ROS 2 (Humble) multi-robot system that demonstrates **Conflict-Based Search (CBS)** with **Space-Time A\*** path planning and an **energy-aware cost model** for coordinating a swarm of 3 autonomous mobile robots in a Gazebo simulation.

---

## Project Architecture

```
mapf_ws/
├── src/
│   ├── mapf_planner/          # CBS + Space-Time A* path planner
│   │   ├── space_time_astar.py    # Low-level (x, y, t) A* search
│   │   ├── cbs.py                 # High-level Conflict-Based Search
│   │   └── planner.py             # ROS 2 node bridging CBS ↔ topics
│   ├── robot_controller/      # Per-robot distributed controllers
│   │   ├── path_follower.py       # Space-time path execution + battery sim
│   │   ├── performance_monitor.py # Fleet metrics aggregation
│   │   ├── live_plotter.py        # Real-time matplotlib dashboard
│   │   └── save_report.py         # PDF/CSV report generator
│   ├── task_allocator/        # Energy-aware swarm task allocation
│   │   └── allocator.py           # Distance + energy + congestion scoring
│   ├── bringup/               # Launch files
│   │   └── launch/full_system.launch.py
│   ├── warehouse_description/ # Robot URDF model
│   │   └── urdf/warehouse_robot.urdf
│   └── warehouse_gazebo/      # Gazebo world + sim launch
│       ├── worlds/mapf_arena.world
│       └── launch/sim.launch.py
└── reports/                   # Generated PDF/CSV reports
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Space-Time A\*** | Path planning in (x, y, t) space with vertex and edge constraints |
| **CBS (Conflict-Based Search)** | Two-level optimal MAPF solver — constraint tree + low-level replanning |
| **Energy-Aware Cost** | `Cost = w_dist × distance + w_turn × turns + w_wait × idle + w_cong × congestion` |
| **Distributed Execution** | Each robot independently follows its conflict-free path |
| **Battery Simulation** | Tracks energy drain from movement, turns, acceleration, and idle time |
| **Swarm Task Allocation** | Assigns goals using energy-distance-congestion heuristic |
| **Collision Avoidance** | CBS guarantees conflict-free paths; near-miss monitoring at runtime |
| **Performance Reporting** | Real-time dashboard + PDF/PNG/CSV report generation |

## Energy-Aware Cost Weights (Default)

| Weight | Value | Description |
|--------|-------|-------------|
| `w_dist` | 1.0 | Distance traveled |
| `w_turn` | 0.3 | Heading changes |
| `w_wait` | 0.2 | Idle/wait actions |
| `w_cong` | 0.4 | Congestion penalty |

---

## Prerequisites

- **Ubuntu 22.04**
- **ROS 2 Humble** (desktop install)
- **Gazebo 11** (comes with `ros-humble-desktop`)
- **Python 3.10+**

Install ROS 2 dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-state-publisher \
  python3-matplotlib \
  python3-numpy
```

---

## Build

```bash
cd ~/mapf_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

For a clean rebuild:

```bash
cd ~/mapf_ws
rm -rf build/ install/ log/
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

---

## Running the Full System

You need **4 terminals**. In each terminal, first run:

```bash
cd ~/mapf_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Terminal 1 — Gazebo Simulation

```bash
ros2 launch warehouse_gazebo sim.launch.py
```

Wait until Gazebo fully opens and you see the 3 robots spawned in the arena.

### Terminal 2 — Full System (Planner + Controllers + Allocator + Monitor)

```bash
ros2 launch bringup full_system.launch.py
```

This launches:
- **CBS Space-Time MAPF Planner** (delay: 12s)
- **3× Path Follower Controllers** (delay: 12s)
- **Swarm Task Allocator** (delay: 18s)
- **Performance Monitor** (delay: 8s)

### Terminal 3 — Live Dashboard (Optional)

```bash
ros2 run robot_controller live_plotter
```

Opens a real-time matplotlib window with 8 panels: battery, distance, velocity, utilization, energy rate, path efficiency, trajectory map, and swarm KPIs.

### Terminal 4 — Generate Report

```bash
ros2 run robot_controller save_report --ros-args -p duration_seconds:=120
```

Collects metrics for 120 seconds, then saves:
- `~/mapf_ws/reports/report_YYYYMMDD_HHMMSS.pdf`
- `~/mapf_ws/reports/report_YYYYMMDD_HHMMSS.png`
- `~/mapf_ws/reports/metrics_YYYYMMDD_HHMMSS.csv`

Change the duration as needed:

```bash
# Quick 60-second report
ros2 run robot_controller save_report --ros-args -p duration_seconds:=60
```

---

## Useful ROS 2 Commands

```bash
# List all active topics
ros2 topic list

# Watch planner stats
ros2 topic echo /mapf_planner/stats

# Watch task assignments
ros2 topic echo /task_assignments

# Watch fleet metrics (JSON)
ros2 topic echo /metrics_json

# Watch a specific robot's battery
ros2 topic echo /robot1/battery_level

# Watch robot status
ros2 topic echo /robot1/status
```

---

## Project Parameters

### CBS Planner (`mapf_planner`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_robots` | 3 | Number of robots in swarm |
| `energy_weight_distance` | 1.0 | Distance cost weight |
| `energy_weight_turns` | 0.3 | Turn penalty weight |
| `energy_weight_wait` | 0.2 | Wait action cost weight |
| `energy_weight_congestion` | 0.4 | Congestion penalty weight |
| `max_planning_time_steps` | 120 | Max time horizon for A* |
| `max_cbs_nodes` | 5000 | Max CBS constraint tree nodes |

### Path Follower (`robot_controller`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_namespace` | `robot1` | Robot namespace |
| `linear_speed` | 0.35 | Max linear velocity (m/s) |
| `battery_capacity` | 100.0 | Starting battery (%) |
| `energy_per_meter` | 0.8 | Battery drain per meter |
| `energy_per_turn` | 0.3 | Battery drain per turn |
| `idle_drain_rate` | 0.01 | Idle battery drain per tick |

### Task Allocator (`task_allocator`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_robots` | 3 | Swarm size |
| `task_interval` | 15.0 | Seconds between task rounds |
| `energy_weight` | 0.4 | Energy factor in scoring |
| `distance_weight` | 0.4 | Distance factor in scoring |
| `congestion_weight` | 0.2 | Congestion factor in scoring |

---

## Algorithm Reference

- **CBS**: Sharon et al., *"Conflict-Based Search for Optimal Multi-Agent Pathfinding,"* Artificial Intelligence, 2015.
- **Space-Time A\***: Each state is `(x, y, t)` on a 4-connected grid + wait action. Supports vertex constraints `(x, y, t)` and edge constraints `(x1, y1) → (x2, y2)` at time `t`.

---

## License

Apache 2.0

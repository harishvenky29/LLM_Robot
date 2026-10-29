"""Nav2 for HomeBot: localisation (AMCL on the pre-made map, or live SLAM)
plus planner, controller, recovery behaviours and the BT navigator.

Normally started by home.launch.py with nav:=nav2 (and slam:=true for SLAM).
Can also be run on its own next to a running simulation:
    ros2 launch llm_home_robot nav2.launch.py slam:=true

Save a SLAM map:  ros2 run nav2_map_server map_saver_cli -f ~/home_map
Use it later:     ros2 launch llm_home_robot home.launch.py nav:=nav2 map:=$HOME/home_map.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NAV_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


def generate_launch_description():
    pkg = get_package_share_directory("llm_home_robot")
    params = LaunchConfiguration("params_file")
    slam = LaunchConfiguration("slam")
    map_yaml = LaunchConfiguration("map")
    sim = {"use_sim_time": True}

    localisation = GroupAction(condition=UnlessCondition(slam), actions=[
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen", parameters=[params, {"yaml_filename": map_yaml}, sim]),
        Node(package="nav2_amcl", executable="amcl", name="amcl", output="screen",
             parameters=[params, sim]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_localization", output="screen",
             parameters=[{"autostart": True, "node_names": ["map_server", "amcl"], **sim}]),
    ])

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")),
        launch_arguments={"slam_params_file": os.path.join(pkg, "config", "slam_params.yaml"),
                          "use_sim_time": "true"}.items(),
        condition=IfCondition(slam),
    )

    navigation = [
        Node(package="nav2_controller", executable="controller_server", output="screen",
             parameters=[params, sim]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server",
             output="screen", parameters=[params, sim]),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server",
             output="screen", parameters=[params, sim]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
             output="screen", parameters=[params, sim]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", output="screen",
             parameters=[{"autostart": True, "node_names": NAV_NODES, **sim}]),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("slam", default_value="false",
                              description="true = build the map live with slam_toolbox"),
        DeclareLaunchArgument("map", default_value=os.path.join(pkg, "maps", "big_home.yaml"),
                              description="map yaml for AMCL (ignored with slam:=true)"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(pkg, "config", "nav2_params.yaml")),
        localisation,
        slam_toolbox,
        *navigation,
    ])

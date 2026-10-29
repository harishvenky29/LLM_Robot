"""VERSION 2 - full robot: everything in version 1 plus camera object search
(YOLO + colour), object memory, follow-me, Nav2 navigation and SLAM mapping.

    ros2 launch llm_home_robot v2_full.launch.py                         # simple route driver (reliable)
    ros2 launch llm_home_robot v2_full.launch.py nav:=nav2               # Nav2 on the ready-made map
    ros2 launch llm_home_robot v2_full.launch.py nav:=nav2 slam:=true    # Nav2 + build the map live
If Nav2 fails to reach a place, the robot falls back to the simple route driver.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg = get_package_share_directory("llm_home_robot")
    pass_through = ["nav", "slam", "llm_model", "llm_backend", "voice", "rviz", "headless", "detector"]
    defaults = {"nav": "graph", "slam": "false", "llm_model": "", "llm_backend": "ollama", "voice": "false",
                "rviz": "true", "headless": "false", "detector": "auto"}
    return LaunchDescription(
        [DeclareLaunchArgument(a, default_value=defaults[a]) for a in pass_through] + [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "home.launch.py")),
                launch_arguments={
                    "perception": "true", "props": "true",
                    **{a: LaunchConfiguration(a) for a in pass_through},
                }.items()),
        ])

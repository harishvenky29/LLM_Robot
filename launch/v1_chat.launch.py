"""VERSION 1 - command robot: chat or voice commands only.

House + robot + place navigation + rule/LLM command understanding.
No camera detection, no downloaded props, no Nav2.

    ros2 launch llm_home_robot v1_chat.launch.py                     # chat (ros2 run llm_home_robot chat)
    ros2 launch llm_home_robot v1_chat.launch.py voice:=true         # + microphone
    ros2 launch llm_home_robot v1_chat.launch.py llm_backend:=none   # rules only
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg = get_package_share_directory("llm_home_robot")
    pass_through = ["llm_model", "llm_backend", "voice", "rviz", "headless"]
    defaults = {"llm_model": "", "llm_backend": "ollama", "voice": "false", "rviz": "true", "headless": "false"}
    return LaunchDescription(
        [DeclareLaunchArgument(a, default_value=defaults[a]) for a in pass_through] + [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "home.launch.py")),
                launch_arguments={
                    "nav": "graph", "perception": "false", "props": "false",
                    **{a: LaunchConfiguration(a) for a in pass_through},
                }.items()),
        ])

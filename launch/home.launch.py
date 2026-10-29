"""Start everything: Gazebo house + HomeBot + bridge + brain (+ Nav2, camera, RViz, voice).

    ros2 launch llm_home_robot home.launch.py                          # graph navigation, Ollama LLM
    ros2 launch llm_home_robot home.launch.py nav:=nav2                # Nav2 + AMCL on the ready-made map
    ros2 launch llm_home_robot home.launch.py nav:=nav2 slam:=true     # Nav2 + live SLAM mapping
    ros2 launch llm_home_robot home.launch.py llm_backend:=none        # rules only, no LLM
    ros2 launch llm_home_robot home.launch.py llm_backend:=anthropic   # Claude (ANTHROPIC_API_KEY)
    ros2 launch llm_home_robot home.launch.py voice:=true              # talk to it (needs a mic)
    ros2 launch llm_home_robot home.launch.py rviz:=false props:=false headless:=true

Then in another terminal:  ros2 run llm_home_robot chat
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def prop_spawners(pkg):
    """One `create` call per Fuel prop. Each is independent, so a failed
    download (offline) only loses that prop, never the world."""
    with open(os.path.join(pkg, "config", "props.yaml")) as f:
        props = yaml.safe_load(f)["props"]
    nodes = []
    for p in props:
        sdf = (f"<sdf version='1.9'><model name='{p['name']}'><static>true</static>"
               f"<include><uri>{p['uri']}</uri></include></model></sdf>")
        nodes.append(Node(package="ros_gz_sim", executable="create", output="log",
                          arguments=["-world", "big_home", "-string", sdf, "-name", p["name"],
                                     "-x", str(p["x"]), "-y", str(p["y"]), "-z", str(p["z"]),
                                     "-Y", str(p["yaw"])]))
    return nodes


def generate_launch_description():
    pkg = get_package_share_directory("llm_home_robot")
    world = os.path.join(pkg, "worlds", "big_home.sdf")
    bridge_cfg = os.path.join(pkg, "config", "bridge.yaml")
    rviz_cfg = os.path.join(pkg, "config", "home.rviz")
    with open(os.path.join(pkg, "urdf", "home_bot.urdf")) as f:
        robot_description = f.read()

    arg = LaunchConfiguration
    brain, rviz, nav, slam = arg("brain"), arg("rviz"), arg("nav"), arg("slam")
    is_nav2 = PythonExpression(["'", nav, "' == 'nav2'"])
    is_graph = PythonExpression(["'", nav, "' != 'nav2'"])
    gz_args = PythonExpression(["'-r ' + ('-s --headless-rendering ' if '", arg("headless"),
                                "' == 'true' else '') + '", world, "'"])
    sim = {"use_sim_time": True}

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "nav2.launch.py")),
        launch_arguments={"slam": slam, "map": arg("map")}.items(),
        condition=IfCondition(is_nav2),
    )

    return LaunchDescription([
        DeclareLaunchArgument("brain", default_value="true", description="start interpreter + executor"),
        DeclareLaunchArgument("rviz", default_value="true", description="open RViz"),
        DeclareLaunchArgument("nav", default_value="graph",
                              description="graph (built-in place-graph driver) | nav2"),
        DeclareLaunchArgument("slam", default_value="false",
                              description="with nav:=nav2, build the map live with slam_toolbox"),
        DeclareLaunchArgument("map", default_value=os.path.join(pkg, "maps", "big_home.yaml"),
                              description="map for AMCL (nav:=nav2 slam:=false)"),
        DeclareLaunchArgument("perception", default_value="true", description="camera object detection"),
        DeclareLaunchArgument("detector", default_value="auto", description="auto | yolo | colour"),
        DeclareLaunchArgument("props", default_value="true",
                              description="spawn people/chairs/cup from Gazebo Fuel (needs internet once)"),
        DeclareLaunchArgument("voice", default_value="false", description="microphone + speech"),
        DeclareLaunchArgument("headless", default_value="false", description="no Gazebo window"),
        DeclareLaunchArgument("llm_backend", default_value="ollama", description="ollama | anthropic | none"),
        DeclareLaunchArgument("llm_model", default_value="",
                              description="model name (default qwen2.5:3b / claude-haiku-5-5)"),
        gazebo,
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description, **sim}], output="screen"),
        Node(package="ros_gz_sim", executable="create", output="screen",
             arguments=["-topic", "robot_description", "-name", "home_bot",
                        "-x", "0.0", "-y", "0.0", "-z", "0.02"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": bridge_cfg, **sim}]),
        # without Nav2 there is no localisation: the map frame is the odom frame
        Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
             arguments=["--frame-id", "map", "--child-frame-id", "odom"],
             parameters=[sim], condition=IfCondition(is_graph)),
        TimerAction(period=4.0, actions=prop_spawners(pkg), condition=IfCondition(arg("props"))),
        nav2,
        Node(package="llm_home_robot", executable="executor", output="screen",
             parameters=[{"nav_backend": nav, **sim}], condition=IfCondition(brain)),
        Node(package="llm_home_robot", executable="interpreter", output="screen",
             parameters=[{"llm_backend": arg("llm_backend"), "llm_model": arg("llm_model"), **sim}],
             condition=IfCondition(brain)),
        Node(package="llm_home_robot", executable="perception", output="screen",
             parameters=[{"detector": arg("detector"), **sim}], condition=IfCondition(arg("perception"))),
        Node(package="llm_home_robot", executable="viz", output="screen",
             parameters=[{"frame_id": "map", **sim}]),
        Node(package="llm_home_robot", executable="voice", output="screen",
             parameters=[sim], condition=IfCondition(arg("voice"))),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg],
             parameters=[sim], output="log", condition=IfCondition(rviz)),
    ])

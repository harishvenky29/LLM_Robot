from glob import glob

from setuptools import setup

package_name = "llm_home_robot"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.rviz")),
        ("share/" + package_name + "/maps", glob("maps/*")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Harish Venkatesan M R",
    maintainer_email="mrharishv29@gmail.com",
    description="Language-commanded home robot in ROS 2 Jazzy + Gazebo Harmonic",
    license="MIT",
    entry_points={
        "console_scripts": [
            "executor = llm_home_robot.executor_node:main",
            "interpreter = llm_home_robot.interpreter_node:main",
            "chat = llm_home_robot.chat_node:main",
            "viz = llm_home_robot.viz_node:main",
            "perception = llm_home_robot.perception_node:main",
            "voice = llm_home_robot.voice_node:main",
            "person_walker = llm_home_robot.person_node:walker_main",
            "person_teleop = llm_home_robot.person_node:teleop_main",
        ],
    },
)

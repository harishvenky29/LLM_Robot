#!/bin/bash
# VIDEO 6 (version 2) - Nav2 + SLAM: the robot maps the house by itself while patrolling.
# Tip: speed the video up 4-8x when editing - the patrol takes a few minutes.
# Usage: bash 6_nav2_slam_patrol.sh [extra launch args]
source "$(dirname "$0")/common.sh"
rm -f ~/.llm_home_robot/objects.yaml
start_sim v2_full.launch.py nav:=nav2 slam:=true "$@"
sleep 10
pause_for_setup

title "LLM Home Robot - Version 2: Nav2 + SLAM patrol (map built live)"
say "patrol the house" --timeout 1200

title "Saving the map to ~/home_map"
ros2 run nav2_map_server map_saver_cli -f ~/home_map --ros-args -p use_sim_time:=true
title "Done! Map saved: ~/home_map.pgm"
sleep 3

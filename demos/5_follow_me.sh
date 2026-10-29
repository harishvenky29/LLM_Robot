#!/bin/bash
# VIDEO 5 (version 2) - follow me. The simulated person walks to the kitchen and the hallway.
# Usage: bash 5_follow_me.sh [extra launch args]
source "$(dirname "$0")/common.sh"
start_sim v2_full.launch.py "$@"
sleep 10
pause_for_setup

title "LLM Home Robot - Version 2: follow me"
say "go to the living room"
say "follow me" --no-wait --pause 4

title "The person walks away - the robot follows"
ros2 run llm_home_robot person_walker --ros-args \
    -p route:="[kitchen, hallway]" -p start_delay:=1.0 -p use_sim_time:=true &
WALKER_PID=$!
wait $WALKER_PID
WALKER_PID=
sleep 5

say "stop following me"
title "Done!"
sleep 3

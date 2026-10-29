#!/bin/bash
# VIDEO 4 (version 2) - camera search: find things, find people, remember where they are.
# Usage: bash 4_find_objects.sh [extra launch args]
source "$(dirname "$0")/common.sh"
rm -f ~/.llm_home_robot/objects.yaml          # start with an empty memory for the video
start_sim v2_full.launch.py "$@"
sleep 10                                      # let Nav2 come up
pause_for_setup

title "LLM Home Robot - Version 2: finding things with the camera"
say "find the red ball"
say "what do you see"

title "Is anyone there?"
say "is anyone in the living room"

title "Search the whole house"
say "find the blue box"

title "It remembers where things are"
say "where is the red ball"

title "Done!"
sleep 3

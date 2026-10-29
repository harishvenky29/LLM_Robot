#!/bin/bash
# VIDEO 7 (version 2) - the LLM with the camera: everyday requests become camera searches,
# following and patrols.
# Needs Ollama (ollama serve + ollama pull qwen2.5:3b).
# Usage: bash 7_llm_planner_camera.sh [extra launch args]
source "$(dirname "$0")/common.sh"
rm -f ~/.llm_home_robot/objects.yaml
start_sim v2_full.launch.py "$@"
pause_for_setup

title "LLM + camera: everyday requests"
say "can you help me find my red ball?"
say "what can you see right now?"
say "I lost my blue box somewhere, could you look for it?"

title "Done!"
sleep 3

#!/bin/bash
# VIDEO 2 (version 1) - talk to the robot. You speak, it moves and answers out loud.
# Needs: pip install faster-whisper sounddevice --break-system-packages
#        sudo apt install libportaudio2 speech-dispatcher
# Usage: bash 2_voice_commands.sh [extra launch args]
source "$(dirname "$0")/common.sh"
start_sim v1_chat.launch.py "$@"
pause_for_setup

title "LLM Home Robot - Version 1: voice commands"
cat <<'CUE'
  Say these one at a time (wait for the robot to finish each):

    1.  "Robot, where are you?"
    2.  "Robot, go to the kitchen."
    3.  "Robot, turn around."
    4.  "Robot, go to the bedroom then come back home."
    5.  "Robot, stop."               (say it while it's moving)

  Ctrl+C when you're done.
CUE
ros2 run llm_home_robot voice --ros-args -p wake_word:=robot

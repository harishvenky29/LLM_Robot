#!/bin/bash
# Shared helpers for the demo scripts. Not run directly.

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/tmp/llm_robot_demo.log

if ! ros2 pkg prefix llm_home_robot >/dev/null 2>&1; then
  echo "Source ROS and your workspace first, e.g.:"
  echo "  source /opt/ros/jazzy/setup.bash && source ~/Projects/LLM_Robot/install/setup.bash"
  exit 1
fi

# warn early if the LLM isn't there (the robot then falls back to the rule parser)
check_llm() {
  if [[ " $* " != *"llm_backend:=none"* && " $* " != *"llm_backend:=anthropic"* ]]; then
    if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
      echo -e "\033[1;91mOllama isn't running - start it with 'ollama serve' (and 'ollama pull qwen2.5:3b').\033[0m"
      echo "Without it the robot uses the rule parser instead of the LLM."
      sleep 3
    fi
  fi
}

# start_sim <launch file> [launch args...]
start_sim() {
  check_llm "$@"
  echo "Starting $1 (log: $LOG) ..."
  ros2 launch llm_home_robot "$@" > "$LOG" 2>&1 &
  SIM_PID=$!
  trap stop_sim EXIT
  trap 'exit 130' INT
  python3 "$DEMO_DIR/send.py" --ready || exit 1
}

stop_sim() {
  [ -n "$WALKER_PID" ] && kill -INT "$WALKER_PID" 2>/dev/null
  if [ -n "$SIM_PID" ]; then
    echo; echo "Shutting down..."
    kill -INT "$SIM_PID" 2>/dev/null
    wait "$SIM_PID" 2>/dev/null
    SIM_PID=
  fi
}

# say "<command>" [send.py options]  - send a command and wait until it's done
say() { python3 "$DEMO_DIR/send.py" "$@"; }

# title "<text>"  - a big caption for the video
title() {
  echo
  echo -e "\033[1;93m==================================================\033[0m"
  echo -e "\033[1;93m  $*\033[0m"
  echo -e "\033[1;93m==================================================\033[0m"
  sleep 2
}

# pause_for_setup  - time to arrange windows and press record
pause_for_setup() {
  echo
  echo -e "\033[1m>>> Arrange the Gazebo / RViz / terminal windows, start recording, then press Enter.\033[0m"
  read -r
}

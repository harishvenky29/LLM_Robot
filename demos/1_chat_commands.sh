#!/bin/bash
# VIDEO 1 (version 1) - basic chat commands: places, motion, sequences, learning a place.
# Usage: bash 1_chat_commands.sh [extra launch args]
source "$(dirname "$0")/common.sh"
start_sim v1_chat.launch.py "$@"
pause_for_setup

title "LLM Home Robot - Version 1: chat commands"
say "what places do you know"
say "where are you"

title "Basic motion"
say "move forward 1 meter"
say "turn left 90 degrees"
say "turn around"

title "Go to a named place"
say "go to the kitchen"
say "take me to the lounge"

title "Several steps in one sentence"
say "go to the bedroom then turn around and come back home"

title "Teach it a new place"
say "go to the office"
say "remember this place as reading corner"
say "go to the dining room"
say "go to reading corner"

title "Done!"
sleep 3

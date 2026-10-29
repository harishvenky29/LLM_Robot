#!/bin/bash
# VIDEO 3 (version 1) - LLM-first planning: the LLM plans every command.
# Needs Ollama:  ollama serve   +   ollama pull qwen2.5:3b
# (or: bash 3_llm_requests.sh llm_backend:=anthropic  with ANTHROPIC_API_KEY set)
# The purple "LLM plan" lines show what the model decided and how long it took.
# Usage: bash 3_llm_requests.sh [extra launch args]
source "$(dirname "$0")/common.sh"
start_sim v1_chat.launch.py "$@"
pause_for_setup

title "LLM Home Robot - the LLM plans every command"
say "I'm really thirsty"

title "Change of mind"
say "actually, I'd rather watch some TV"

title "Multi-step plans"
say "go to the bedroom, wait 3 seconds, then come back to the hallway"

title "Understands what you mean, not just keywords"
say "I need to answer some emails"
say "someone is ringing the doorbell"

title "Knows its limits"
say "what's the weather like in York?"
say "go to the garden"

title "Done!"
sleep 3

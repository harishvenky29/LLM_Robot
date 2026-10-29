# Recording the demo videos

Each script starts the simulation, waits for you to press **Enter** (so you can arrange windows and start recording), then sends the commands by itself and prints them as subtitles in the terminal. When it finishes it shuts everything down.

## The two versions

| | Launch file | What's on |
|---|---|---|
| **Version 1 – command robot** | `ros2 launch llm_home_robot v1_chat.launch.py` | house, robot, place navigation, chat + voice + LLM commands |
| **Version 2 – full robot** | `ros2 launch llm_home_robot v2_full.launch.py` | everything in v1 + camera object search (YOLO), object memory, follow-me, Nav2, SLAM |

## The videos

| # | Script | Version | Shows | Length |
|---|---|---|---|---|
| 1 | `1_chat_commands.sh` | v1 | places, motion, sequences, teaching a place | ~3 min |
| 2 | `2_voice_commands.sh` | v1 | **you speak**, it moves and answers aloud | ~2 min |
| 3 | `3_llm_requests.sh` | v1 | the LLM: indirect requests, multi-step, knowing its limits | ~3 min |
| 4 | `4_find_objects.sh` | v2 | find the red ball, a person, the blue box; memory | ~4 min |
| 5 | `5_follow_me.sh` | v2 | robot follows the walking person | ~2 min |
| 6 | `6_nav2_slam_patrol.sh` | v2 | patrols every room while building the map | ~5 min (speed up) |
| 7 | `7_llm_planner_camera.sh` | v2 | LLM + camera: everyday requests become camera searches | ~3 min |

All scripts use the **LLM planner** (start Ollama first: `ollama serve`, and once `ollama pull qwen2.5:3b`). The purple 🧠 lines in the terminal show each plan, who made it (LLM or rule parser) and how long it took. Add `llm_backend:=none` to any script to run it with the rule parser only.

## How to record one

```bash
source /opt/ros/jazzy/setup.bash
source ~/Projects/LLM_Robot/install/setup.bash
cd ~/Projects/LLM_Robot/llm_home_robot/demos

bash 1_chat_commands.sh
```

1. Wait for **"Robot ready"** and the Gazebo + RViz windows.
2. Arrange the windows: RViz on one side, Gazebo on the other, the terminal at the bottom (the subtitles).
3. Start recording: **Ctrl+Shift+Alt+R** (Ubuntu's built-in screen recorder; press again to stop, saved to `~/Videos/Screencasts`) or use OBS.
4. Press **Enter** in the terminal. Stop recording when it says "Done!".

Tips:
- Close other programs; the first run of version 2 downloads the people/chair models and YOLO weights. Do a practice run first.
- In Gazebo, right-click the robot → **Follow** for a camera that tracks it.
- In RViz, the camera panel (bottom left) is great for video 4.
- Commands that take long (patrol) look better sped up when editing.

## Turning a recording into a GIF for GitHub

```bash
sudo apt install ffmpeg
# 4x faster, 960 px wide, 12 fps
ffmpeg -i ~/Videos/Screencasts/video4.webm -vf "setpts=0.25*PTS,fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" docs/find_objects.gif
# or a small MP4 (GitHub plays MP4s dragged into a README/issue in the web editor)
ffmpeg -i ~/Videos/Screencasts/video4.webm -vf "setpts=0.5*PTS,scale=1280:-2" -an -c:v libx264 -crf 28 docs/find_objects.mp4
```

Keep GIFs under ~10 MB. Then in the main `README.md`:

```markdown
![Finding objects](docs/find_objects.gif)
```

## Your own commands

`say` in the scripts is just `python3 send.py "<command>"` — run it in a second terminal any time the robot is running:

```bash
python3 send.py "go to the office"
```

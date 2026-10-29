# llm_home_robot — a voice- and language-commanded home robot (ROS 2 Jazzy + Gazebo Harmonic)

Talk to a robot in plain English — *"go to the kitchen"*, *"find the red ball"*, *"follow me"*, *"is anyone in the office?"*, *"I'm thirsty"* — and watch it navigate a 7-room simulated house, search rooms with its camera, and follow a person around.

![house map and simulated tour](tour.png)

| What the robot's camera sees during "find the red ball" | …and during "find someone" (YOLO) |
|---|---|
| ![red ball](camera_red_ball.png) | ![person](camera_person.png) |

## Two versions

| | Start it with | What you get |
|---|---|---|
| **Version 1 – command robot** | `ros2 launch llm_home_robot v1_chat.launch.py` | house, robot, place navigation; chat, voice and LLM commands (go to, move, turn, sequences, learn places, follow me, patrol) |
| **Version 2 – full robot** | `ros2 launch llm_home_robot v2_full.launch.py` | everything in v1 + camera object search and object memory; Nav2 and SLAM with `nav:=nav2` (falls back to the simple driver if Nav2 fails) |

Then talk to it: `ros2 run llm_home_robot chat` (type) or `ros2 run llm_home_robot voice` (speak).

## Demo videos

Scripted demos you can run and screen-record are in [`demos/`](../demos/README.md): chat commands, voice, LLM requests (v1), finding objects, follow-me, and a Nav2 + SLAM patrol (v2).

<!-- add your recorded GIFs here, e.g.  ![Finding objects](docs/find_objects.gif) -->

## Features

| | |
|---|---|
| **House + robot** | 20 × 14 m Gazebo house (living room, kitchen, dining room, hallway, bedroom, bathroom, office) with furniture, four coloured objects, real people/chairs/toilet/cup models from Gazebo Fuel, and a person you can walk around. *HomeBot* is a differential-drive robot with a 360° lidar and an RGB camera on a mast. |
| **Knows the house** | 14 named places with aliases ("lounge", "loo", "fridge"), room boundaries and a doorway graph (`config/places.yaml`). Learns new places on request. |
| **LLM** | Every command goes to an LLM (qwen2.5 or llama3.2 running locally via Ollama, or Claude), which turns it into robot actions: "I'm thirsty" → go to the fridge, "go to the bedroom, wait 5 s and come back" → three steps. The actions are validated before the robot moves; a rule parser is the backup. |
| **Two navigation stacks** | `nav:=graph` — built-in doorway-to-doorway driver, zero extra dependencies. `nav:=nav2` — Nav2 (A* planner, Regulated Pure Pursuit, costmaps, recoveries) with AMCL on a ready-made map, or **live SLAM** with slam_toolbox. |
| **Sees things** | Camera → YOLO11 (person, chair, couch, tv, cup, bottle, toilet, …) + a colour detector. Detections are placed on the map with lidar range + size check and remembered (semantic map), so *"where is the cup?"* works later. |
| **Finds things** | *"find the cup"* → checks memory, then searches the likeliest rooms first (cup → kitchen, dining room), turns a full circle in each, and drives up to the object when it sees it. |
| **Follows you** | Lidar leg/body tracker that ignores everything on the map, so it locks onto *you*, not the sofa. Keeps ~0.9 m behind, through doorways. |
| **Voice** | Whisper (faster-whisper, runs locally) with voice-activity detection or push-to-talk, optional wake word ("robot, …"), and spoken replies. |
| **RViz** | House, room and place labels, route graph, planned path, Nav2/SLAM map, camera view with detection boxes, remembered objects, the person being followed, robot status and speech bubble. Click a place to send the robot there. |

## Architecture

```
  🎤 voice ─┐                                   ┌─────────── camera ──────────┐
            ▼                                   ▼                             │
 you ──▶ /robot/command ──▶ interpreter ──▶ /robot/actions ──▶ executor ◀── perception
 (chat)                     1) rules         [{"type":"find",    │  ▲        (YOLO + colour)
                            2) LLM → JSON      "object":"cup"}]  │  │ /robot/detections
                               → validate()                      │  │
                                                                 ▼  │
               graph driver  ──or──  Nav2 (NavigateToPose) ◀─────┤  ├── /scan (safety stop,
                                       AMCL / slam_toolbox       │  │    follow-me tracker,
                                                                 ▼  │    object range)
                                                     /cmd_vel → Gazebo Harmonic
                                         /robot/reply ──▶ chat, voice (spoken), RViz
                                         /robot/status, /robot/objects ──▶ RViz markers
```

## How the LLM is used

```
 "I'm thirsty"  ──▶  LLM (qwen2.5:3b via Ollama, or Claude)
                      prompt: the robot's actions + the house's places + current room
                ◀──  {"reply": "Heading to the fridge.", "actions": [{"type": "goto", "place": "fridge"}]}
                      │
                 validate()   known action types and places only, distances/angles clamped
                      │
                      ▼
                  executor  →  drives the robot
```

- **Every command goes to the LLM.** The rule parser is only a backup: it's used if the LLM is not running or returns something invalid, and for `stop`, which never waits for the LLM.
- The LLM can never move the robot directly — it can only return actions from a fixed list, and `validate()` rejects unknown places or actions.
- `llm_backend:=none` turns the LLM off (rule parser only).

Check the LLM on its own (no simulator needed): `python3 tools/eval_llm.py` sends 17 typical requests and checks the actions.

| Model (Ollama) | Score | Missed |
|---|---|---|
| **qwen2.5:3b** (default) | 15 / 17 | "I need to answer some emails" (declined instead of going to the desk) · "keep me company while I walk around" (stayed put) |
| llama3.2:3b | 15 / 17 | "…then turn right" (turned left) · first request timed out while the model was loading |

About 6 s per request with qwen2.5:3b on the 2-CPU, no-GPU machine used for testing; a normal laptop is faster.

## Install (Ubuntu 24.04, ROS 2 Jazzy)

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-robot-state-publisher ros-jazzy-navigation2 \
                 ros-jazzy-slam-toolbox python3-yaml python3-opencv python3-pip
cd ~/Projects/LLM_Robot          # any colcon workspace; this folder goes inside it
colcon build --packages-select llm_home_robot --symlink-install
source install/setup.bash
```

Optional extras (each one is skipped gracefully if missing):

```bash
# YOLO object detection (CPU-only PyTorch keeps the download small)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install ultralytics "numpy<2" --break-system-packages

# Voice: speech-to-text + text-to-speech
pip install faster-whisper sounddevice --break-system-packages
sudo apt install libportaudio2 speech-dispatcher

# Local LLM for free-form requests
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:3b
```

> Keep `numpy<2`: ROS Jazzy's Python packages are built against numpy 1.x.

## Run

```bash
# terminal 1
ros2 launch llm_home_robot home.launch.py                        # graph navigation + camera + RViz
# terminal 2
ros2 run llm_home_robot chat
```

| Launch option | |
|---|---|
| `nav:=nav2` | Nav2 + AMCL on the ready-made map (`maps/big_home.yaml`) |
| `nav:=nav2 slam:=true` | Nav2 + slam_toolbox — builds the map as it drives (try `patrol the house`) |
| `llm_model:=llama3.2:3b` | another Ollama model (default `qwen2.5:3b`; `qwen2.5:7b` if your machine is fast) |
| `llm_backend:=none` / `anthropic` | rules only / Claude (`export ANTHROPIC_API_KEY=...`) — default `ollama` |
| `voice:=true` | start the microphone node in this terminal |
| `detector:=colour` | colour detector only (no YOLO) |
| `props:=false` | don't download the Fuel people/chairs (offline) |
| `perception:=false` `rviz:=false` `headless:=true` | lighter runs |

**Voice in its own terminal** (recommended — you see what it heard):

```bash
ros2 run llm_home_robot voice                                     # hands-free
ros2 run llm_home_robot voice --ros-args -p wake_word:=robot      # "robot, go to the kitchen"
ros2 run llm_home_robot voice --ros-args -p mode:=ptt             # press Enter to talk
ros2 run llm_home_robot voice --ros-args -p wav:=command.wav      # no mic? test with a recording
```

Questions (*where are you*, *what do you see*, *where is the cup*) are answered without interrupting what the robot is doing; any other command replaces the current one.

**Make the person move** (for follow-me):

```bash
ros2 run llm_home_robot person_walker                                         # kitchen → living room → hallway
ros2 run llm_home_robot person_walker --ros-args -p route:="[bedroom, office]" -p loop:=true
ros2 run llm_home_robot person_teleop                                         # drive with w/a/s/d
```

**Save a SLAM map and reuse it:**

```bash
ros2 run nav2_map_server map_saver_cli -f ~/home_map
ros2 launch llm_home_robot home.launch.py nav:=nav2 map:=$HOME/home_map.yaml
```

## Commands

| Type | Examples |
|---|---|
| Motion | `move forward` · `go forward 2 meters` · `back up 50 cm` · `turn left` · `turn right 45 degrees` · `turn around` |
| Places | `go to the kitchen` · `take me to the loo` · `head to the lounge` · `go home` |
| Find | `find the red ball` · `look for my mug` · `find someone` · `is anyone in the kitchen?` · `is there a cup in the dining room?` |
| Memory | `where is the cup?` · `have you seen my phone?` · `what do you see?` |
| People | `follow me` · `come with me` · `stop following me` |
| House | `patrol the house` / `explore` (visits every room, catalogues objects, maps with SLAM) |
| Sequences | `go to the bedroom then find the blue box` |
| Info | `where are you` · `what places do you know` |
| Control | `stop` · `wait 5 seconds` |
| Learn | `remember this place as reading corner` → later `go to reading corner` |
| Needs the LLM | `I'm thirsty` → fridge · `I need to answer some emails` → desk · `someone's at the door` → front door · `I want to watch TV` → sofa · `what's the weather?` → politely declines |

Things to find in the house: **red ball** (living room), **blue box** (bedroom), **green bin** (office), **yellow bucket** (dining room) — always detectable; plus YOLO-recognisable **people** (living room, bedroom, office), **chairs**, a **toilet**, a **cup** on the kitchen island and a **Coke can** on the dining table.

## RViz

![RViz preview](rviz_preview.png)

| What you see | Meaning |
|---|---|
| Blue pin + white label | a known place, with aliases; arrow = heading on arrival |
| Green pin | the current goal · Purple pin: a place the robot learned |
| Orange line | the remaining route (graph mode) · Green line: the Nav2 plan |
| Coloured cubes + labels | objects the robot has seen and where (fade after 10 min) |
| Cyan column "YOU" | the person being followed |
| Label above the robot | state, goal, "looking for …", room, position; "sees: …" below it |
| Camera panel | live camera with YOLO (yellow) and colour (blue) boxes |
| Grey map | the Nav2 / SLAM occupancy grid |

**Click to go:** pick **Publish Point** in the toolbar and click near a pin. With Nav2, **2D Pose Estimate** re-localises the robot.

## Topics

| Topic | Type | |
|---|---|---|
| `/robot/command` | `std_msgs/String` | text in (chat, voice, RViz click) |
| `/robot/actions` | `std_msgs/String` JSON | interpreter → executor |
| `/robot/reply` | `std_msgs/String` | what the robot says |
| `/robot/status` | `std_msgs/String` JSON | state, room, pose, goal, route, following, sees — 5 Hz |
| `/robot/detections` | `std_msgs/String` JSON | perception → executor: label, conf, bbox, bearing |
| `/robot/objects` | `std_msgs/String` JSON | remembered objects (also saved to `~/.llm_home_robot/objects.yaml`) |
| `/camera/image`, `/camera/detections` | `sensor_msgs/Image` | raw / annotated camera |
| `/person/cmd_vel`, `/person/odom` | Twist / Odometry | the walkable person |
| `/cmd_vel` `/odom` `/scan` `/tf` `/map` `/plan` | standard | Gazebo bridge, Nav2, SLAM |

## Customising the house

1. Edit walls, furniture, coloured objects, Fuel props and the person's start in `llm_home_robot/house_layout.py` (shared by Gazebo, the Nav2 map, RViz, the follow-me filter and the tests); places and routes in `config/places.yaml`.
2. `python3 tools/generate_world.py` regenerates `worlds/big_home.sdf`, `maps/big_home.pgm/.yaml` and `config/props.yaml`, and checks every place and route is ≥ 0.28 m from anything (it names the edge that collides).
3. Rebuild.

## Testing

```bash
python3 -m pytest test/ -v           # 112 tests, no ROS needed: parser, routing, perception,
                                     # colour detector, object memory, follow tracker, voice VAD, map
python3 tools/offline_sim.py --plot  # headless tour of all 14 places with simulated lidar
```

![map built by SLAM during "patrol the house"](slam_map.png)

*The map slam_toolbox built while the robot patrolled the house on its own (`nav:=nav2 slam:=true`, then `patrol the house`) — walls, doorways, furniture, chair legs and people.*

Everything was also run in Gazebo Harmonic + ROS 2 Jazzy (headless): finding the red ball (colour) and a person (YOLO, 0.89 conf), following the walking person from the living room through the side door, the kitchen and into the hallway at 0.9–1.5 m, Nav2 + AMCL goals, a SLAM patrol of the whole house, and spoken commands through VAD → Whisper → parser.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Package 'llm_home_robot' not found` | `source install/setup.bash` in every new terminal |
| No people/chairs in Gazebo | first launch downloads them from Gazebo Fuel — needs internet once; or `props:=false` |
| "YOLO unavailable" in the log | `pip install ultralytics "numpy<2" --break-system-packages` (colour objects still work) |
| ROS errors mentioning numpy 2 | `pip install "numpy<2" --break-system-packages` |
| "Nav2 isn't running yet" | wait ~10 s after launch for the lifecycle managers to activate |
| Robot follows the wrong thing | stand within 3 m in open floor, then say `follow me` again |
| Voice hears nothing / too much | `-p min_rms:=0.02` (noisy room) or `-p device:=N` (`python3 -m sounddevice` lists mics) |
| Slow simulation | `perception:=false`, `rviz:=false`, or `detector:=colour` |

## Layout

```
llm_home_robot/
├── config/  places.yaml, bridge.yaml, nav2_params.yaml, slam_params.yaml, props.yaml, home.rviz
├── launch/  home.launch.py (everything), nav2.launch.py (localisation/SLAM + Nav2)
├── maps/    big_home.pgm/.yaml (generated from the layout)
├── llm_home_robot/
│   ├── house_layout.py      # single source of geometry
│   ├── place_graph.py       # rooms, aliases, Dijkstra
│   ├── command_parser.py    # rules + validate()
│   ├── llm_client.py        # Ollama / Claude
│   ├── interpreter_node.py  # text -> actions
│   ├── executor_node.py     # actions -> motion (graph / Nav2), find, follow, patrol, memory
│   ├── perception.py        # detectors, camera geometry, object memory (pure Python)
│   ├── perception_node.py   # camera -> /robot/detections
│   ├── follow.py            # lidar person tracker (pure Python)
│   ├── voice_utils.py       # VAD, transcript clean-up, TTS (pure Python)
│   ├── voice_node.py        # mic -> Whisper -> /robot/command, replies -> speech
│   ├── person_node.py       # person_walker / person_teleop
│   ├── viz_markers.py, viz_node.py   # RViz
│   └── chat_node.py
├── urdf/home_bot.urdf       # diff drive + lidar + camera
├── worlds/big_home.sdf      # generated
├── tools/  generate_world.py, offline_sim.py, preview_markers.py
└── test/   test_brain.py, test_viz.py, test_upgrades.py
```

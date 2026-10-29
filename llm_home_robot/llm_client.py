"""Simple LLM client: text in, list of robot actions out.

Backends (stdlib only - no extra pip installs):
  * ollama    - local, free. `ollama pull qwen2.5:3b` (or llama3.2:3b)
  * anthropic - Claude API, needs ANTHROPIC_API_KEY in the environment

The LLM never drives the motors: it only returns JSON actions, which
command_parser.validate() checks before the robot does anything.
"""
import json
import os
import re
import urllib.request

PROMPT = """You control a small wheeled home robot. Turn the user's request into robot actions.
Reply with ONLY a JSON object: {{"reply": "<one short sentence>", "actions": [ ... ]}}

Places (use the name before the brackets): {places}
The robot is in the {room}.

Actions:
  {{"type": "goto", "place": "<place>"}}
  {{"type": "move", "distance": <metres, negative = backwards>}}
  {{"type": "turn", "angle": <degrees, left = positive, right = negative>}}
  {{"type": "wait", "seconds": <number>}}
  {{"type": "stop"}}
  {{"type": "where"}}
  {{"type": "save_place", "name": "<new name>"}}
  {{"type": "follow"}}                        follow the person near the robot (walk with me, come with me)
  {{"type": "patrol"}}                        visit every room and report everything seen
{camera}
Rules:
- Indirect requests: thirsty -> fridge, tired -> bed, work -> desk, battery low -> charging_station,
  doorbell -> front_door, watch TV -> sofa.
- Home = charging_station. "Go home", "go back home", "dock" -> goto charging_station.
- Questions ("where are you?", "how far is...?", "what can you do?") -> answer in "reply";
  use "where" for the robot's position. Questions never move the robot.
- The text may come from speech recognition: "petrol" means patrol, "follow knee" means follow me.
- Several steps -> several actions in order.
- If you can't help, use "actions": [] and explain in "reply".

Examples:
"I'm thirsty" -> {{"reply": "Heading to the fridge.", "actions": [{{"type": "goto", "place": "fridge"}}]}}
"back up a bit and turn right" -> {{"reply": "Backing up and turning right.", "actions": [{{"type": "move", "distance": -0.5}}, {{"type": "turn", "angle": -90}}]}}
"what's the weather today?" -> {{"reply": "Sorry, I can't check the weather - I can only move around the house.", "actions": []}}
"go for a petrol" -> {{"reply": "Starting a patrol of the house.", "actions": [{{"type": "patrol"}}]}}"""

CAMERA = """  {{"type": "find", "object": "<thing>", "room": "<room, only if the user names one>"}}
                                              search with the camera ("person" for people)
  {{"type": "locate", "object": "<thing>"}}   say where it was last seen
  {{"type": "look"}}                          say what the camera sees
  Camera examples:
  "is anyone in the dining room?" -> [{{"type": "find", "object": "person", "room": "dining_room"}}]
  "how many people are in the house?" -> [{{"type": "patrol"}}]  (the patrol report counts people and objects)
"""


def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON in LLM output: {text[:200]!r}")
        return json.loads(m.group(0))


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class LLMClient:
    def __init__(self, backend="ollama", model=None, url=None, timeout=120.0):
        self.backend, self.timeout = backend, timeout
        if backend == "ollama":
            self.model = model or "qwen2.5:3b"
            self.url = (url or "http://localhost:11434").rstrip("/") + "/api/chat"
        elif backend == "anthropic":
            self.model = model or "claude-haiku-5-5"
            self.url = url or "https://api.anthropic.com/v1/messages"
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            raise ValueError(f"unknown LLM backend {backend!r}")

    def interpret(self, text, places, room, camera=False):
        """Returns (actions, reply)."""
        system = PROMPT.format(places=", ".join(places), room=(room or "house").replace("_", " "),
                               camera=CAMERA.format() if camera else "")
        if self.backend == "ollama":
            out = _post(self.url, {
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_predict": 300},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            }, {}, self.timeout)
            content = out["message"]["content"]
        else:
            if not self.api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            out = _post(self.url, {"model": self.model, "max_tokens": 400, "system": system,
                                   "messages": [{"role": "user", "content": text}]},
                        {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
            content = "".join(b.get("text", "") for b in out.get("content", []))
        data = extract_json(content)
        actions = data.get("actions", [])
        return (actions if isinstance(actions, list) else [actions]), str(data.get("reply", "")).strip()

"""Interpreter node: what you say -> robot actions.

  sub  /robot/command  std_msgs/String   e.g. "I'm thirsty"
  sub  /robot/status   std_msgs/String   current room + whether the camera is on
  pub  /robot/actions  std_msgs/String   JSON {"actions": [...], "source": "llm" | "rules"}
  pub  /robot/plan     std_msgs/String   JSON {text, source, actions, seconds} (shown by the demos)
  pub  /robot/reply    std_msgs/String   what the robot says back

Simple LLM-first flow:
  1. "stop" is handled immediately (never waits for the LLM).
  2. Everything else goes to the LLM, which returns a list of actions.
  3. The actions are checked by validate() (known actions/places, clamped values).
  4. If the LLM is unavailable or returns something invalid, the rule parser
     handles the command instead.
Set llm_backend:=none to use the rule parser only.
"""
import json
import os
import re
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from .command_parser import describe, parse, validate
from .llm_client import LLMClient
from .place_graph import PlaceGraph

LEARNED_FILE = os.path.expanduser("~/.llm_home_robot/learned_places.yaml")
QUIET = ("where", "list_places", "look", "locate", "stop", "forget")
STOP_RE = re.compile(r"^\s*(please\s+)?(stop|halt|freeze|abort|cancel)\b", re.I)


class Interpreter(Node):
    def __init__(self):
        super().__init__("command_interpreter")
        share = get_package_share_directory("llm_home_robot")
        p = self.declare_parameter
        p("places_file", os.path.join(share, "config", "places.yaml"))
        p("learned_file", LEARNED_FILE)
        p("llm_backend", "ollama")       # ollama | anthropic | none
        p("llm_model", "")
        p("llm_url", "")
        p("llm_timeout", 120.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.places_file, self.learned_file = gp("places_file"), gp("learned_file")
        self.llm = None
        if gp("llm_backend") != "none":
            try:
                self.llm = LLMClient(gp("llm_backend"), gp("llm_model") or None, gp("llm_url") or None,
                                     gp("llm_timeout"))
            except ValueError as e:
                self.get_logger().error(str(e))
        self.get_logger().info(f"LLM: {self.llm.backend + ' / ' + self.llm.model if self.llm else 'none (rules only)'}")
        self.status = {}
        self.generation = 0              # a newer command makes an older LLM answer stale
        self.lock = threading.Lock()
        self.pub_actions = self.create_publisher(String, "/robot/actions", 10)
        self.pub_reply = self.create_publisher(String, "/robot/reply", 10)
        self.pub_plan = self.create_publisher(String, "/robot/plan", 10)
        self.create_subscription(String, "/robot/command", self.on_command, 10)
        self.create_subscription(String, "/robot/status", self.on_status, 10)
        self.get_logger().info("Ready. Publish text on /robot/command (or run `ros2 run llm_home_robot chat`).")

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def say(self, text):
        self.get_logger().info(f"robot: {text}")
        self.pub_reply.publish(String(data=text))

    def run(self, text, actions, source, reply=None, seconds=0.0):
        self.pub_plan.publish(String(data=json.dumps(
            {"text": text, "source": source, "actions": actions, "seconds": round(seconds, 1)})))
        if actions:
            self.pub_actions.publish(String(data=json.dumps({"actions": actions, "source": source})))
        if reply:
            self.say(reply)
        elif actions and actions[0]["type"] not in QUIET:
            self.say(f"OK - {describe(actions)}.")

    def on_command(self, msg):
        text = msg.data.strip()
        if not text:
            return
        self.generation += 1
        self.get_logger().info(f"user: {text}")
        graph = PlaceGraph.from_files(self.places_file, self.learned_file)
        if self.llm is None or STOP_RE.match(text):
            self.use_rules(text, graph)
            return
        threading.Thread(target=self.ask_llm, args=(text, graph, self.generation), daemon=True).start()

    def use_rules(self, text, graph):
        actions = parse(text, graph)
        if actions:
            self.run(text, actions, "rules")
        else:
            self.run(text, [], "rules")
            self.say("Sorry, I didn't understand. Try e.g. 'go to the kitchen' or 'turn left 45 degrees'.")

    def ask_llm(self, text, graph, gen):
        with self.lock:
            if gen != self.generation:
                return
            self.say("Thinking...")
            t0 = time.time()
            try:
                raw, reply = self.llm.interpret(text, place_list(graph), self.status.get("room"),
                                                camera=bool(self.status.get("camera")))
                actions = validate(raw, graph) if raw else []
            except Exception as e:  # noqa: BLE001  (LLM down, bad JSON, invalid plan)
                self.get_logger().warn(f"LLM failed ({e}) - using the rule parser")
                if gen == self.generation:
                    self.use_rules(text, graph)
                return
            if gen != self.generation:
                return                   # a newer command (or stop) arrived meanwhile
            self.run(text, actions, "llm", reply or ("I can't do that." if not actions else None),
                     time.time() - t0)


def place_list(graph):
    """'charging_station (home, dock)' - nicknames help the LLM understand 'go home'."""
    out = []
    for name in graph.place_names():
        al = [a for a in graph.places[name]["aliases"] if a != name.replace("_", " ")][:3]
        out.append(f"{name} ({', '.join(al)})" if al else name)
    return out


def main():
    rclpy.init()
    node = Interpreter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

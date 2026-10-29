#!/usr/bin/env python3
"""Send one command to the robot, print it and the robot's replies (nice for
recording), and wait until the robot has finished.

    python3 send.py "go to the kitchen"            # wait until done
    python3 send.py --ready                        # just wait until the robot is up
    python3 send.py "follow me" --no-wait
"""
import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

BOLD, CYAN, GREEN, DIM, END = "\033[1m", "\033[96m", "\033[92m", "\033[2m", "\033[0m"


MAGENTA = "\033[95m"


def show_plan(m):
    """Print how the command was understood: by the LLM (with timing) or the rules."""
    try:
        p = json.loads(m.data)
    except json.JSONDecodeError:
        return
    src = p.get("source", "?")
    who = {"llm": "LLM", "rules": "rule parser"}.get(src, src)
    t = f", {p['seconds']:.1f}s" if p.get("seconds") else ""
    steps = " -> ".join(step(a) for a in p.get("actions", [])) or "(answer only)"
    print(f"   {MAGENTA}🧠 {who}{t}: {steps}{END}", flush=True)


def step(a):
    t = a["type"]
    if t == "goto":
        return f"goto {a['place']}"
    if t == "find":
        s = f"find {a['object']}" + (f" in {a['room']}" if a.get("room") else "")
        if a.get("if_found"):
            s += " [if found: " + ", ".join(step(b) for b in a["if_found"]) + "]"
        if a.get("if_not_found"):
            s += " [else: " + ", ".join(step(b) for b in a["if_not_found"]) + "]"
        return s
    if t == "move":
        return f"move {a['distance']:+.1f}m"
    if t == "turn":
        return f"turn {a['angle']:+.0f}°"
    if t == "say":
        return f'say "{a["text"]}"'
    if t == "wait":
        return f"wait {a['seconds']:.0f}s"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?")
    ap.add_argument("--ready", action="store_true", help="wait for the robot to start")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the command to finish")
    ap.add_argument("--timeout", type=float, default=400.0)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to linger afterwards")
    a = ap.parse_args()

    rclpy.init()
    node = Node("demo_send")
    state = {"status": None, "busy_seen": False, "plan_t": None}

    def on_status(m):
        try:
            state["status"] = json.loads(m.data)
        except json.JSONDecodeError:
            return
        if state["status"].get("state") != "idle":
            state["busy_seen"] = True

    node.create_subscription(String, "/robot/status", on_status, 10)
    node.create_subscription(
        String, "/robot/reply",
        lambda m: print(f"   {GREEN}🤖 {m.data}{END}", flush=True), 10)
    def on_plan(m):
        state["plan_t"] = time.time()
        show_plan(m)

    node.create_subscription(String, "/robot/plan", on_plan, 10)
    pub = node.create_publisher(String, "/robot/command", 10)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    if a.ready:
        print(f"{DIM}Waiting for the robot to start...{END}", flush=True)
        t0 = time.time()
        while state["status"] is None and time.time() - t0 < 180:
            spin(0.5)
        if state["status"] is None:
            print("Robot didn't start - check the launch log.")
            sys.exit(1)
        print(f"{DIM}Robot ready.{END}", flush=True)
        spin(1.0)
        return

    t0 = time.time()                              # connect to the robot before talking
    while (pub.get_subscription_count() == 0 or node.count_publishers("/robot/reply") == 0
           or state["status"] is None) and time.time() - t0 < 30:
        spin(0.2)
    spin(0.5)                                     # let the reply subscription finish matching
    print(f"\n{BOLD}{CYAN}🧑 you> {a.command}{END}", flush=True)
    pub.publish(String(data=a.command))
    if a.no_wait:
        spin(a.pause)
        return

    t0 = time.time()
    while time.time() - t0 < a.timeout:
        spin(0.2)
        st = state["status"] or {}
        idle = st.get("state") == "idle" and not st.get("queued")
        if state["busy_seen"] and idle:
            break
        planned = state["plan_t"] is not None
        if planned and not state["busy_seen"] and time.time() - state["plan_t"] > 4.0:
            break                                  # answer only / instant action (where are you, ...)
        if not planned and time.time() - t0 > max(180.0, a.timeout):
            print("   (no plan after 3 minutes - is the LLM very slow?)")
            break
    spin(a.pause)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

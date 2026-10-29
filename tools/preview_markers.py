#!/usr/bin/env python3
"""Draw the RViz markers top-down with matplotlib (no ROS needed).

    python3 tools/preview_markers.py      -> docs/rviz_preview.png

Uses the exact marker dicts viz_node publishes, with the robot mid-way to the
kitchen, so you can check labels and layout without launching anything.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llm_home_robot import viz_markers as vm  # noqa: E402
from llm_home_robot.place_graph import PlaceGraph  # noqa: E402


def main():
    g = PlaceGraph.from_files(os.path.join(ROOT, "config", "places.yaml"))
    g.add_place("reading corner", -8.8, 2.0)           # show a learned place
    status = {"state": "busy", "goal": "kitchen", "room": "hallway", "looking_for": "cup",
              "x": -2.5, "y": 0.0, "yaw_deg": 0, "sees": [],
              "route": [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [1.0, 3.5]]}
    # what a patrol leaves in the robot's object memory
    objects = [{"label": lab, "x": x, "y": y, "seen": 0} for lab, x, y in [
        ("red ball", -8.3, 2.8), ("blue box", -8.6, -1.8), ("green bin", 2.2, -6.3),
        ("yellow bucket", 9.2, 6.2), ("person", -9.1, -5.6), ("person", 8.6, -2.3),
        ("chair", 6.0, 5.2), ("chair", 8.0, -5.8), ("toilet", -2.45, -5.0), ("bottle", 7.9, 5.3)]]
    pins = [(q["x"], q["y"]) for q in g.places.values()]
    ms = (vm.house_markers(g.rooms, places=pins) + vm.graph_markers(g)
          + vm.place_markers(g, "kitchen", {"reading_corner"})
          + vm.robot_markers(status, "Looking for a cup. Checking the kitchen...")
          + vm.object_markers(objects, now=0))

    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.patch.set_facecolor((38 / 255, 41 / 255, 46 / 255))
    ax.set_facecolor((38 / 255, 41 / 255, 46 / 255))
    order = ["floors", "route_graph", "waypoints", "walls", "furniture", "place_pins",
             "place_headings", "objects", "robot_ring", "planned_route", "furniture_names",
             "room_names", "place_names", "object_names", "robot_status", "speech"]
    for ns in order:
        for m in (m for m in ms if m["ns"] == ns):
            c = m["rgba"]
            if c[3] == 0:
                continue
            t, x, y = m["type"], m["x"], m["y"]
            if t == "cube":
                ax.add_patch(Rectangle((x - m["sx"] / 2, y - m["sy"] / 2), m["sx"], m["sy"],
                                       color=c[:3], alpha=c[3], lw=0, zorder=order.index(ns)))
            elif t == "cylinder":
                ax.add_patch(Circle((x, y), m["sx"] / 2, color=c[:3], alpha=c[3], zorder=order.index(ns)))
            elif t == "arrow":
                yaw = m.get("yaw", 0)
                ax.add_patch(FancyArrow(x, y, m["sx"] * math.cos(yaw), m["sx"] * math.sin(yaw),
                                        width=0.04, head_width=0.14, color=c[:3], alpha=c[3],
                                        length_includes_head=True, zorder=order.index(ns)))
            elif t == "line_list":
                p = m["points"]
                for a, b in zip(p[::2], p[1::2]):
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=c[:3], alpha=c[3], lw=1, ls="--",
                            zorder=order.index(ns))
            elif t == "line_strip":
                ax.plot([p[0] for p in m["points"]], [p[1] for p in m["points"]], color=c[:3],
                        lw=4, alpha=c[3], zorder=order.index(ns), solid_capstyle="round")
            elif t == "text":
                if ns == "furniture_names":
                    continue                     # hidden by default in home.rviz
                dy = {"robot_status": 0.75, "speech": 1.55, "object_names": -0.35}.get(ns, 0)
                size = {"room_names": 13, "robot_status": 10, "speech": 9.5, "object_names": 7.5}.get(ns, 8)
                bbox = dict(boxstyle="round,pad=0.3", fc=(0.1, 0.1, 0.1, 0.75), ec="none") \
                    if ns in ("speech", "robot_status") else None
                ax.text(x, y + dy, m["text"], ha="center", va="center", fontsize=size,
                        color=c[:3], weight="bold" if ns != "place_names" else "normal",
                        zorder=20, bbox=bbox, linespacing=1.1)
    # the robot body itself (drawn by RViz's RobotModel display)
    ax.add_patch(Rectangle((status["x"] - 0.2, status["y"] - 0.15), 0.4, 0.3,
                           color=(0.95, 0.45, 0.1), zorder=15))
    ax.set_aspect("equal")
    ax.set_xlim(-10.6, 10.6)
    ax.set_ylim(-7.4, 7.6)
    ax.axis("off")
    out = os.path.join(ROOT, "docs", "rviz_preview.png")
    fig.savefig(out, dpi=105, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()

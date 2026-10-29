#!/usr/bin/env python3
"""Headless check of the executor WITHOUT ROS or Gazebo.

Mocks rclpy, simulates a differential-drive robot with a ray-cast lidar
against the exact walls/furniture from generate_world.py, then sends the
robot on a tour of every place and reports clearance and success.

    python3 tools/offline_sim.py            # tour all places
    python3 tools/offline_sim.py --plot     # also write tour.png (needs matplotlib)
"""
import json
import math
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


# ----------------------------------------------------------- fake ROS 2 API
class _Clock:
    t = 0.0

    def now(self):
        return types.SimpleNamespace(nanoseconds=int(_Clock.t * 1e9))


class _Pub:
    def __init__(self, topic):
        self.topic, self.msgs = topic, []

    def publish(self, msg):
        self.msgs.append(msg)


class _Logger:
    def info(self, m): print(f"  [{_Clock.t:6.1f}s] {m}")
    def warn(self, m): print(f"  [warn] {m}")
    error = warn


class _Node:
    def __init__(self, name):
        self._params, self.pubs, self.subs = {}, {}, {}

    def declare_parameter(self, n, v):
        self._params[n] = v

    def get_parameter(self, n):
        return types.SimpleNamespace(value=self._params[n])

    def create_publisher(self, _t, topic, _q):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]

    def create_subscription(self, _t, topic, cb, _q):
        self.subs[topic] = cb

    def create_timer(self, *_):
        pass

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return _Logger()


def _msg(**kw):
    return lambda **k: types.SimpleNamespace(**{**kw, **k})


class Twist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0)
        self.angular = types.SimpleNamespace(z=0.0)


mods = {
    "rclpy": types.SimpleNamespace(init=lambda: None, ok=lambda: True, spin=None, shutdown=None),
    "rclpy.node": types.SimpleNamespace(Node=_Node),
    "ament_index_python": types.SimpleNamespace(),
    "ament_index_python.packages": types.SimpleNamespace(get_package_share_directory=lambda _p: ROOT),
    "geometry_msgs": types.SimpleNamespace(), "geometry_msgs.msg": types.SimpleNamespace(Twist=Twist),
    "nav_msgs": types.SimpleNamespace(), "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
    "sensor_msgs": types.SimpleNamespace(), "sensor_msgs.msg": types.SimpleNamespace(LaserScan=object),
    "std_msgs": types.SimpleNamespace(), "std_msgs.msg": types.SimpleNamespace(String=_msg(data="")),
}
sys.modules.update(mods)

import generate_world as gw  # noqa: E402
from llm_home_robot.executor_node import Executor  # noqa: E402

OBS = gw.obstacles()
BODY_R = 0.25          # half-diagonal of the 0.40 x 0.30 body
DT = 0.05


def ray(x, y, a, max_r=10.0):
    """Distance to the nearest box along a ray (slab method)."""
    dx, dy = math.cos(a), math.sin(a)
    best = max_r
    for _n, cx, cy, sx, sy in OBS:
        tmin, tmax = -math.inf, math.inf
        for o, d, c, s in ((x, dx, cx, sx), (y, dy, cy, sy)):
            lo, hi = c - s / 2, c + s / 2
            if abs(d) < 1e-9:
                if not lo <= o <= hi:
                    tmin = math.inf
                    break
            else:
                t1, t2 = (lo - o) / d, (hi - o) / d
                tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        if tmin <= tmax and 0 < tmin < best:
            best = tmin
    return best


def clearance(x, y):
    return min(gw.point_box_dist(x, y, *o[1:]) for o in OBS) - BODY_R


def run(commands, max_time=400.0, plot=None):
    _Clock.t = 0.0
    ex = Executor()
    x, y, yaw = 0.0, 0.0, 0.0
    trail, min_clear, worst = [], math.inf, None
    n_rays = 72
    for cmd in commands:
        ex.subs["/robot/actions"](types.SimpleNamespace(data=json.dumps({"actions": cmd})))
        print(f"\n> {cmd}")
        started = _Clock.t
        while True:
            q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
            ex.on_odom(types.SimpleNamespace(pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y), orientation=q))))
            if int(_Clock.t / DT) % 2 == 0:  # 10 Hz lidar
                lx, ly = x + 0.05 * math.cos(yaw), y + 0.05 * math.sin(yaw)
                ranges = [ray(lx, ly, yaw - math.pi + i * 2 * math.pi / n_rays) for i in range(n_rays)]
                ex.on_scan(types.SimpleNamespace(ranges=ranges, range_min=0.12, range_max=10.0,
                                                 angle_min=-math.pi, angle_increment=2 * math.pi / n_rays))
            ex.step()
            tw = ex.pubs["/cmd_vel"].msgs[-1] if ex.pubs["/cmd_vel"].msgs else Twist()
            v, w = tw.linear.x, tw.angular.z
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw = math.atan2(math.sin(yaw + w * DT), math.cos(yaw + w * DT))
            _Clock.t += DT
            trail.append((x, y))
            c = clearance(x, y)
            if c < min_clear:
                min_clear, worst = c, (round(x, 2), round(y, 2))
            if ex.task is None and not ex.queue:
                break
            if _Clock.t - started > max_time:
                print("  !! TIMEOUT")
                return False, min_clear, worst, trail
    return True, min_clear, worst, trail


def main():
    from llm_home_robot.place_graph import PlaceGraph
    g = PlaceGraph.from_files(os.path.join(ROOT, "config", "places.yaml"))
    tour = ["kitchen", "fridge", "dining_table", "office", "desk", "bathroom", "bedroom", "bed",
            "living_room", "sofa", "kitchen", "front_door", "charging_station", "hallway"]
    cmds = [[{"type": "goto", "place": p}] for p in tour]
    cmds += [[{"type": "move", "distance": 1.5}, {"type": "turn", "angle": 90},
              {"type": "turn", "angle": -450}, {"type": "move", "distance": -1.0}, {"type": "where"}]]
    ok, min_clear, worst, trail = run(cmds)
    print(f"\nRESULT: {'all commands completed' if ok else 'FAILED'}; "
          f"closest approach to any obstacle = {min_clear:.2f} m at {worst}")
    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(12, 8.6))
        for name, (x0, y0, x1, y1) in g.rooms.items():
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=gw.FLOOR_RGB[name], alpha=0.5, lw=0))
            ax.text((x0 + x1) / 2, y1 - 0.45 if name != "hallway" else 0.55, name.replace("_", " "),
                    ha="center", fontsize=10, weight="bold", color="#333")
        for n, cx, cy, sx, sy in OBS:
            ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                                   color="#555" if n == "wall" else "#b08850"))
        for a, nbrs in g.adj.items():
            for b in nbrs:
                ax.plot(*zip(g.nodes[a][:2], g.nodes[b][:2]), color="#999", lw=0.6, ls="--", zorder=1)
        ax.plot(*zip(*trail), color="#e8590c", lw=1.4, zorder=3, label="robot path")
        for p, d in g.places.items():
            ax.plot(d["x"], d["y"], "o", color="#1c7ed6", ms=6, zorder=4)
            ax.annotate(p, (d["x"], d["y"]), xytext=(4, -10), textcoords="offset points", fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_title("HomeBot house: rooms, place graph and simulated tour")
        ax.legend(loc="lower right")
        out = os.path.join(ROOT, "docs", "tour.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print("wrote", out)
    sys.exit(0 if ok and min_clear > 0 else 1)


if __name__ == "__main__":
    main()

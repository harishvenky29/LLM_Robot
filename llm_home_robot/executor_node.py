"""Executor node: runs action lists on the robot.

  sub  /robot/actions     std_msgs/String   JSON from the interpreter
  sub  /odom              nav_msgs/Odometry
  sub  /scan              sensor_msgs/LaserScan   (safety stop, follow-me, object range)
  sub  /robot/detections  std_msgs/String   JSON from perception_node (camera)
  pub  /cmd_vel           geometry_msgs/Twist
  pub  /robot/status      std_msgs/String   JSON {state, room, x, y, yaw, goal, route, ...} 5 Hz
  pub  /robot/objects     std_msgs/String   JSON list of remembered objects (semantic map)
  pub  /robot/reply       std_msgs/String   spoken-style feedback ("Arrived at the kitchen")

Two navigation backends (parameter nav_backend):
  graph  - follows the place graph doorway-to-doorway with a P-controller (no extra deps)
  nav2   - sends NavigateToPose goals to Nav2 (AMCL or SLAM map), with costmap
           obstacle avoidance and replanning

Behaviours: move, turn, goto, wait, stop, where, list_places, save_place,
follow (lidar person tracking), find (search rooms + camera, then approach),
look (what can you see), locate (where did you see X), forget (clear the
object memory), patrol (visit every
room and catalogue objects - also maps the house when SLAM is running).

A new command pre-empts whatever the robot was doing.
"""
import itertools
import json
import math
import os
import time
from collections import deque

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from . import follow as fl
from . import house_layout as house
from . import perception as pc
from .place_graph import PlaceGraph

DATA_DIR = os.path.expanduser("~/.llm_home_robot")
LEARNED_FILE = os.path.join(DATA_DIR, "learned_places.yaml")
OBJECTS_FILE = os.path.join(DATA_DIR, "objects.yaml")
INFO_ACTIONS = {"where", "look", "locate", "list_places", "forget"}
HALF_WIDTH = 0.23        # robot half width incl. margin, for the safety corridor
_ids = itertools.count(1)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Executor(Node):
    def __init__(self):
        super().__init__("robot_executor")
        share = get_package_share_directory("llm_home_robot")
        p = self.declare_parameter
        p("places_file", os.path.join(share, "config", "places.yaml"))
        p("learned_file", LEARNED_FILE)
        p("objects_file", OBJECTS_FILE)
        p("nav_backend", "graph")    # graph | nav2
        p("map_frame", "map")
        p("base_frame", "base_footprint")
        p("max_linear", 0.4)         # m/s
        p("max_angular", 1.0)        # rad/s
        p("stop_distance", 0.45)     # m from robot centre to obstacle in the driving corridor
        p("blocked_timeout", 8.0)    # s, give up after being blocked this long
        p("waypoint_tolerance", 0.15)
        p("goal_tolerance", 0.08)
        p("follow_distance", 0.9)    # m to keep behind the person
        p("min_confidence", 0.45)    # detections below this are ignored
        p("approach_distance", 0.8)  # m to stop in front of a found object
        p("scan_speed", 0.5)         # rad/s when looking around a room
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.places_file, self.learned_file = gp("places_file"), gp("learned_file")
        self.objects_file = gp("objects_file")
        self.backend = gp("nav_backend")
        self.map_frame, self.base_frame = gp("map_frame"), gp("base_frame")
        self.vmax, self.wmax = gp("max_linear"), gp("max_angular")
        self.stop_dist, self.blocked_timeout = gp("stop_distance"), gp("blocked_timeout")
        self.wp_tol, self.goal_tol = gp("waypoint_tolerance"), gp("goal_tolerance")
        self.follow_dist, self.min_conf = gp("follow_distance"), gp("min_confidence")
        self.approach_dist, self.scan_speed = gp("approach_distance"), gp("scan_speed")

        self.graph = PlaceGraph.from_files(self.places_file, self.learned_file)
        self.odom_pose = None            # (x, y, yaw) in odom
        self.odom_hist = deque(maxlen=60)  # (stamp, pose) ~3 s, to undo camera latency
        self.map_pose = None             # (x, y, yaw) in map (TF), if available
        self.scan = None                 # (ranges, angle_min, angle_inc, t)
        self.scan_seq = 0
        self.front_min = math.inf
        self.rear_min = math.inf
        self.dets, self.dets_t, self.camera_seen = [], -1e9, False
        self.memory = pc.ObjectMemory(merge_dist=1.5)
        self.memory_dirty, self.memory_saved_t = False, 0.0
        self.load_memory()
        self.queue = deque()
        self.task = None
        self.blocked_since = None
        self.nav_plan = []
        self.find_branches = {}          # find id -> (if_found, if_not_found, approach)
        # things that are on the map can't be the person we follow
        self.static_obs = [(0.2 if b[0] == "wall" else 0.3, b) for b in house.obstacles()]

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/robot/status", 10)
        self.pub_objects = self.create_publisher(String, "/robot/objects", 10)
        self.pub_reply = self.create_publisher(String, "/robot/reply", 10)
        self.create_subscription(String, "/robot/actions", self.on_actions, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(String, "/robot/detections", self.on_detections, 10)
        self.setup_tf()
        if self.backend == "nav2":
            self.setup_nav2()
        self.create_timer(0.05, self.step)            # 20 Hz control loop
        self.create_timer(0.2, self.publish_status)   # 5 Hz status
        self.create_timer(1.0, self.publish_objects)  # 1 Hz object memory
        self.get_logger().info(f"Executor ready ({self.backend} navigation). "
                               f"Known places: {', '.join(self.graph.place_names())}")

    # ------------------------------------------------------------ plumbing
    def setup_tf(self):
        try:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        except ImportError:          # offline simulator / no tf2
            self.tf_buffer = None

    def setup_nav2(self):
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import Path
        from rclpy.action import ActionClient
        self.NavigateToPose = NavigateToPose
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.create_subscription(Path, "/plan", self.on_plan, 5)
        self.get_logger().info("Using Nav2 (navigate_to_pose) for goto.")

    @property
    def pose(self):
        """Best pose for 'where am I' questions: map frame if TF has it."""
        return self.map_pose or self.odom_pose

    def update_map_pose(self):
        if self.tf_buffer is None:
            return
        try:
            from rclpy.time import Time
            tr = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:  # noqa: BLE001  (not available yet / no map frame)
            return
        q = tr.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.map_pose = (tr.transform.translation.x, tr.transform.translation.y, yaw)

    # ------------------------------------------------------------ callbacks
    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        st = msg.header.stamp
        self.odom_hist.append((st.sec + st.nanosec * 1e-9, self.odom_pose))

    def on_scan(self, msg):
        front, rear = math.inf, math.inf
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            a = msg.angle_min + i * msg.angle_increment
            x, y = 0.05 + r * math.cos(a), r * math.sin(a)     # base frame
            if abs(y) < HALF_WIDTH:
                if x > 0:
                    front = min(front, x)
                else:
                    rear = min(rear, -x)
        self.front_min, self.rear_min = front, rear
        self.scan = (list(msg.ranges), msg.angle_min, msg.angle_increment, self.now())
        self.scan_seq += 1

    def on_detections(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.camera_seen = True
        self.dets = [d for d in data.get("detections", []) if d.get("conf", 0) >= self.min_conf]
        self.dets_t = self.now()
        if self.pose is None or self.scan is None:
            return
        kind = (self.task or {}).get("kind")
        if kind not in (None, "scan", "approach", "wait"):
            return                       # only map objects while deliberately looking, not while driving
        then = self.pose_at(data.get("stamp"))
        if then is None:
            return
        dyaw = wrap(self.odom_pose[2] - then[2])
        if math.dist(then[:2], self.odom_pose[:2]) > 0.25:
            return                       # moved too much since the photo - don't trust positions
        for d in self.dets:
            if not pc.rememberable(d["label"]) or (d.get("source") == "yolo" and d["conf"] < 0.5):
                continue
            bearing_now = wrap(d["bearing"] - dyaw)         # where it is relative to us now
            est = pc.estimate_range(d["label"], d["bbox"], data.get("height", 480), data.get("fx"))
            rng = pc.fuse_range(self.det_range(dict(d, bearing=bearing_now)), est)
            if rng is None or not 0.35 < rng < 6.0:
                continue
            d["range"] = round(rng, 2)
            x, y = pc.to_world(self.pose, bearing_now, rng)
            self.memory.add(d["label"], round(x, 2), round(y, 2), d["conf"], self.graph.room_at(x, y))
            self.memory_dirty = True

    def pose_at(self, stamp):
        """Odom pose closest to a timestamp (the camera image time)."""
        if not self.odom_hist:
            return None
        if stamp is None:
            return self.odom_pose
        best = min(self.odom_hist, key=lambda tp: abs(tp[0] - stamp))
        return best[1] if abs(best[0] - stamp) < 1.0 else self.odom_pose

    def on_plan(self, msg):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.nav_plan = pts[::max(1, len(pts) // 25)] + pts[-1:] if pts else []

    def on_actions(self, msg):
        try:
            actions = json.loads(msg.data)["actions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.get_logger().error(f"bad actions message: {msg.data}")
            return
        if actions and self.odom_pose is not None and all(a.get("type") in INFO_ACTIONS for a in actions):
            for a in actions:            # questions don't interrupt what the robot is doing
                self.start(a)
            return
        self.cancel_nav()
        self.find_branches.clear()
        self.queue = deque(actions)      # a new command replaces the current plan
        self.task = None
        self.blocked_since = None
        self.drive(0.0, 0.0)

    # --------------------------------------------------------------- helpers
    def say(self, text):
        self.get_logger().info(text)
        self.pub_reply.publish(String(data=text))

    def drive(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub_cmd.publish(t)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def room(self):
        return self.graph.room_at(*self.pose[:2]) if self.pose else None

    def nice(self, name):
        return (name or "").replace("_", " ")

    def det_range(self, d):
        if self.scan is None:
            return None
        ranges, amin, ainc, _t = self.scan
        return pc.range_at_bearing(ranges, amin, ainc, d["bearing"])

    def person_candidates(self):
        """Person-sized lidar blobs (base frame) that aren't known furniture/objects."""
        ranges, amin, ainc, _ = self.scan
        out = []
        for c in fl.candidates(fl.scan_points(ranges, amin, ainc)):
            wx, wy = fl.base_to_odom(self.pose, c)            # map frame
            if self.graph.room_at(wx, wy) is None:
                continue
            if any(house.point_box_dist(wx, wy, *b[1:]) < m for m, b in self.static_obs):
                continue
            out.append(c)
        return out

    def visible(self, query, max_age=0.8):
        """Freshest matching detection, or None."""
        if self.now() - self.dets_t > max_age:
            return None
        hits = [d for d in self.dets if query == "*" or pc.matches(query, d["label"])]
        return max(hits, key=lambda d: d["conf"]) if hits else None

    def load_memory(self):
        try:
            with open(self.objects_file) as f:
                self.memory.load((yaml.safe_load(f) or {}).get("objects"))
        except (OSError, yaml.YAMLError):
            pass

    def save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.objects_file), exist_ok=True)
            with open(self.objects_file, "w") as f:
                yaml.safe_dump({"objects": self.memory.to_list()}, f, sort_keys=False)
            self.memory_dirty = False
        except OSError as e:
            self.get_logger().warn(f"could not save objects: {e}")

    def publish_objects(self):
        self.pub_objects.publish(String(data=json.dumps(self.memory.to_list())))
        if self.memory_dirty and time.time() - self.memory_saved_t > 5.0:
            self.memory_saved_t = time.time()
            self.save_memory()

    def publish_status(self):
        if self.pose is None:
            return
        x, y, yaw = self.pose
        t = self.task or {}
        route = []
        if t.get("kind") == "goto":
            route = [[round(px, 2), round(py, 2)] for px, py in t["pts"]]
        elif t.get("kind") == "goto_nav2":
            route = [[round(px, 2), round(py, 2)] for px, py in self.nav_plan]
        status = {
            "state": "idle" if not t and not self.queue else ("blocked" if self.blocked_since else "busy"),
            "task": t.get("kind"),
            "goal": t.get("place"),
            "looking_for": t.get("query") if t.get("query") not in (None, "*") else None,
            "following": t.get("kind") == "follow",
            "person": self.person_in_map(t),
            "queued": len(self.queue),
            "room": self.room(),
            "nearest_place": self.graph.nearest_place(x, y),
            "x": round(x, 2), "y": round(y, 2), "yaw_deg": round(math.degrees(yaw)),
            "route": route,
            "nav": self.backend,
            "camera": self.camera_seen,
            "sees": sorted({d["label"] for d in self.dets}) if self.now() - self.dets_t < 1.0 else [],
        }
        self.pub_status.publish(String(data=json.dumps(status)))

    def person_in_map(self, t):
        tr = t.get("tracker") if t.get("kind") == "follow" else None
        if not tr or tr.target is None:
            return None
        rel = fl.odom_to_base(self.odom_pose, tr.target)      # tracker works in odom
        return [round(v, 2) for v in fl.base_to_odom(self.pose, rel)]

    # ------------------------------------------------------------ main loop
    def step(self):
        self.update_map_pose()
        if self.odom_pose is None:
            return
        if self.task is None:
            if not self.queue:
                return
            self.start(self.queue.popleft())
            if self.task is None:      # instant action (where/stop/...) finished
                return
        done = getattr(self, f"run_{self.task['kind']}")()
        if done:
            if self.task and self.task.get("kind") != "goto_nav2":
                self.drive(0.0, 0.0)
            self.task = None
            self.blocked_since = None
            if not self.queue:
                self.publish_status()

    def start(self, a):
        x, y, yaw = self.odom_pose
        kind = a["type"]
        if kind == "move":
            self.task = {"kind": "move", "start": (x, y), "yaw0": yaw, "dist": float(a["distance"])}
        elif kind == "turn":
            self.task = {"kind": "turn", "left": math.radians(float(a["angle"])), "last": yaw}
        elif kind == "goto":
            self.start_goto(a["place"], quiet=a.get("quiet", False))
        elif kind == "wait":
            self.task = {"kind": "wait", "until": self.now() + float(a["seconds"])}
        elif kind == "stop":
            self.queue.clear()
            self.cancel_nav()
            self.drive(0.0, 0.0)
            self.say("Stopped.")
        elif kind == "where":
            px, py, pyaw = self.pose
            place = self.graph.nearest_place(px, py)
            self.say(f"I'm in the {self.nice(self.room() or 'unknown area')}, near the "
                     f"{self.nice(place)} (x={px:.1f}, y={py:.1f}, facing {math.degrees(pyaw):.0f}°).")
        elif kind == "list_places":
            self.say("I know: " + ", ".join(self.nice(p) for p in self.graph.place_names()) + ".")
        elif kind == "save_place":
            self.save_place(a["name"])
        elif kind == "follow":
            self.task = {"kind": "follow", "tracker": fl.PersonTracker(), "phase": "acquire",
                         "spun": 0.0, "last": yaw, "seq": -1, "announced": False}
        elif kind == "find":
            self.start_find(a["object"], a.get("room"), a.get("if_found") or [],
                            a.get("if_not_found") or [], a.get("approach", True))
        elif kind == "scan":
            self.task = {"kind": "scan", "query": a["object"], "fid": a.get("fid"),
                         "left": 2 * math.pi, "last": yaw, "final": a.get("final", False),
                         "give_up": a.get("give_up")}
        elif kind == "approach":
            self.task = {"kind": "approach", "query": a["object"], "start": (x, y),
                         "seen_t": self.now(), "bearing": a.get("bearing", 0.0)}
        elif kind == "look":
            self.say_look()
        elif kind == "forget":
            self.memory.items.clear()
            self.save_memory()
            self.say("OK, I've forgotten everything I've seen.")
        elif kind == "locate":
            self.say_locate(a["object"])
        elif kind == "patrol":
            self.start_patrol()
        elif kind == "say":
            if a["text"] == "PATROL_SUMMARY":
                self.save_memory()
                self.say("Patrol finished. " + self.memory.summary(self.nice))
            else:
                self.say(a["text"])
        else:
            self.get_logger().warn(f"unknown action {a}")

    # ------------------------------------------------------------- goto
    def route_rooms(self, names):
        rooms = []
        for n in names:
            r = self.graph.nodes[n][2]
            if r and (not rooms or rooms[-1] != r):
                rooms.append(r)
        return rooms

    def start_goto(self, place, quiet=False):
        x, y, _ = self.pose
        if place not in self.graph.places:
            self.say(f"I don't know where the {self.nice(place)} is.")
            return
        plan = self.graph.plan(x, y, place)
        if plan is None and self.backend != "nav2":
            self.say(f"I can't find a route to the {self.nice(place)}.")
            self.queue.clear()
            return
        if plan:
            names, pts, yaw = plan
            length = math.dist((x, y), pts[0]) + self.graph.path_length(names)
            if not quiet:
                self.say(f"Going to the {self.nice(place)} via "
                         f"{' -> '.join(self.nice(r) for r in self.route_rooms(names))} ({length:.1f} m).")
        else:
            pts, yaw = [], self.graph.places[place]["yaw"]
            if not quiet:
                self.say(f"Going to the {self.nice(place)}.")
        if self.backend == "nav2":
            self.start_nav2_goal(place, yaw, quiet)
        else:
            self.task = {"kind": "goto", "place": place, "pts": deque(pts), "yaw": yaw, "quiet": quiet}

    def run_goto(self):
        t = self.task
        x, y, yaw = self.odom_pose
        if t["pts"]:
            gx, gy = t["pts"][0]
            dist = math.hypot(gx - x, gy - y)
            final = len(t["pts"]) == 1
            if dist < (self.goal_tol if final else self.wp_tol):
                t["pts"].popleft()
                return False
            err = wrap(math.atan2(gy - y, gx - x) - yaw)
            if abs(err) > 0.5:                      # face the waypoint first
                self.blocked_since = None
                w = clamp(2.0 * err, -self.wmax, self.wmax)
                self.drive(0.0, w if abs(w) > 0.2 else math.copysign(0.2, w))
                return False
            if self.blocked(True):
                return self.task.get("kind") == "abort"
            v = clamp(0.8 * dist, 0.06, self.vmax) * max(math.cos(err), 0.0)
            w = clamp(2.5 * err, -self.wmax, self.wmax)
            self.drive(v, w)
            return False
        if t["yaw"] is not None:                    # face the place's preferred heading
            err = wrap(t["yaw"] - yaw)
            if abs(err) > 0.03:
                w = clamp(2.0 * err, -self.wmax, self.wmax)
                self.drive(0.0, w if abs(w) > 0.15 else math.copysign(0.15, w))
                return False
        if not t.get("quiet"):
            self.say(f"Arrived at the {self.nice(t['place'])}.")
        return True

    # -- Nav2 backend
    def start_nav2_goal(self, place, yaw, quiet=False):
        p = self.graph.places[place]
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.say("Nav2 isn't running yet - give it a few seconds after launch and try again.")
            self.queue.clear()
            return
        goal = self.NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = p["x"], p["y"]
        yaw = yaw if yaw is not None else 0.0
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        token = next(_ids)
        self.nav_plan = []
        self.task = {"kind": "goto_nav2", "place": place, "token": token, "handle": None,
                     "result": None, "quiet": quiet}
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(lambda f, tok=token: self.on_nav_accepted(f, tok))

    def on_nav_accepted(self, fut, token):
        t = self.task
        handle = fut.result()
        if not t or t.get("token") != token:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            t["result"] = "rejected"
            return
        t["handle"] = handle
        handle.get_result_async().add_done_callback(lambda f, tok=token: self.on_nav_result(f, tok))

    def on_nav_result(self, fut, token):
        t = self.task
        if t and t.get("token") == token:
            status = fut.result().status
            t["result"] = "ok" if status == 4 else ("canceled" if status == 5 else "failed")

    def run_goto_nav2(self):
        t = self.task
        if t["result"] is None:
            return False
        if t["result"] == "ok":
            if not t.get("quiet"):
                self.say(f"Arrived at the {self.nice(t['place'])}.")
        elif t["result"] != "canceled":
            # Nav2 gave up (planner/controller failure): drive the doorway route myself instead
            x, y, _ = self.pose
            plan = self.graph.plan(x, y, t["place"])
            if plan:
                self.get_logger().warn(f"Nav2 failed ({t['result']}) - using the simple route driver")
                self.say(f"Nav2 couldn't plan that, so I'll take my usual route to the {self.nice(t['place'])}.")
                _names, pts, yaw = plan
                self.task = {"kind": "goto", "place": t["place"], "pts": deque(pts), "yaw": yaw,
                             "quiet": t.get("quiet")}
                return False
            self.say(f"I couldn't get to the {self.nice(t['place'])}.")
            self.queue.clear()
        return True

    def cancel_nav(self):
        t = self.task
        if t and t.get("kind") == "goto_nav2":
            t["token"] = None
            if t.get("handle") is not None:
                t["handle"].cancel_goal_async()

    # ------------------------------------------------------------- other actions
    def save_place(self, name):
        x, y, yaw = self.pose
        os.makedirs(os.path.dirname(self.learned_file), exist_ok=True)
        data = {}
        if os.path.exists(self.learned_file):
            with open(self.learned_file) as f:
                data = yaml.safe_load(f) or {}
        canonical = self.graph.add_place(name, x, y, math.degrees(yaw), connect=True)
        data.setdefault("places", {})[canonical] = {
            "x": round(x, 3), "y": round(y, 3), "yaw": round(math.degrees(yaw), 1)}
        with open(self.learned_file, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True)
        self.say(f"Got it - I'll remember this spot in the {self.nice(self.room() or 'house')} "
                 f"as '{self.nice(canonical)}'.")

    def say_look(self):
        if not self.camera_seen:
            self.say("My camera isn't running - launch with perception:=true.")
            return
        if self.now() - self.dets_t > 1.5 or not self.dets:
            self.say("I don't see anything I recognise right now.")
            return
        parts = []
        for d in sorted(self.dets, key=lambda d: -d["conf"])[:5]:
            side = "ahead" if abs(d["bearing"]) < 0.2 else ("on my left" if d["bearing"] > 0 else "on my right")
            dist = f" about {d['range']:.1f} m away" if d.get("range") else ""
            parts.append(f"a {d['label']} {side}{dist}")
        self.say("I can see " + ", ".join(parts) + ".")

    def say_locate(self, query):
        o = self.memory.find(query, min_count=3)
        if not o:
            self.say(f"I haven't seen a {query} yet. Say 'find the {query}' and I'll look for it.")
            return
        ago = time.time() - o["seen"]
        when = "just now" if ago < 60 else (f"{ago / 60:.0f} min ago" if ago < 3600 else "a while ago")
        near = self.graph.nearest_place(o["x"], o["y"])
        self.say(f"I saw a {o['label']} in the {self.nice(o['room'] or 'house')} near the "
                 f"{self.nice(near)} ({when}).")

    def rooms_by_distance(self):
        x, y, _ = self.pose
        rooms = [r for r in self.graph.rooms if r in self.graph.places]
        start = self.graph.nearest_node(x, y, same_room=True)

        def dist(r):
            path = self.graph.shortest_path(start, r)
            return self.graph.path_length(path) if path else math.inf
        return sorted(rooms, key=dist)

    def found_it(self, fid, query, bearing):
        """Object seen: drop the rest of the search, then approach and/or run
        the if_found branch (LLM-first conditional plans)."""
        then, _else, approach = self.find_branches.pop(fid, ([], [], True))
        self.queue = deque(a for a in self.queue if a.get("fid") != fid or fid is None)
        steps = ([{"type": "approach", "object": query, "bearing": bearing}] if approach else []) + list(then)
        self.queue.extendleft(reversed(steps))

    def not_found(self, fid):
        _then, other, _a = self.find_branches.pop(fid, ([], [], True))
        self.queue.extendleft(reversed(list(other)))

    def start_find(self, query, room=None, if_found=(), if_not_found=(), approach=True):
        if not self.camera_seen:
            self.say("I need my camera to look for things - launch with perception:=true.")
            return
        fid = next(_ids)
        self.find_branches[fid] = (list(if_found), list(if_not_found), approach)
        if room:                                  # "is anyone in the kitchen?"
            steps = [] if self.room() == room else [{"type": "goto", "place": room, "quiet": True, "fid": fid}]
            what = "anyone" if query == "person" else f"a {query}"
            self.say(f"Let me check the {self.nice(room)} for {what}.")
            steps.append({"type": "scan", "object": query, "fid": fid, "final": True,
                          "give_up": f"No, I can't see {what} in the {self.nice(room)}."})
            self.queue.extendleft(reversed(steps))
            return
        d = self.visible(query)
        if d:
            self.say(f"I can see a {d['label']} right here.")
            self.found_it(fid, query, d["bearing"])
            return
        steps = []
        o = self.memory.find(query, min_count=3)
        here = self.room()
        if o:
            spot = self.graph.nearest_place(o["x"], o["y"])
            self.say(f"I saw a {o['label']} in the {self.nice(o['room'])} earlier - going to check.")
            steps += [{"type": "goto", "place": spot, "quiet": True},
                      {"type": "scan", "object": query, "fid": fid}]
        else:
            self.say(f"Looking for a {query}. I'll start here, then check the rooms where it's likely to be.")
        steps.append({"type": "scan", "object": query, "fid": fid})        # look around here first
        rooms = pc.search_rooms(query, [r for r in self.rooms_by_distance() if r != here])
        for r in rooms:
            steps += [{"type": "goto", "place": r, "quiet": True, "fid": fid},
                      {"type": "say", "text": f"Checking the {self.nice(r)}...", "fid": fid},
                      {"type": "scan", "object": query, "fid": fid}]
        steps[-1]["final"] = True
        steps[-1]["give_up"] = f"I searched every room but couldn't find a {query}."
        # move the "checking" message before the goto so it's said on the way
        for i in range(len(steps) - 1):
            if steps[i]["type"] == "goto" and steps[i + 1]["type"] == "say":
                steps[i], steps[i + 1] = steps[i + 1], steps[i]
        self.queue.extendleft(reversed(steps))

    def run_scan(self):
        """Turn a full circle looking for the object (or cataloguing everything
        when query == '*')."""
        t = self.task
        q = t["query"]
        if q != "*":
            d = self.visible(q)
            if d:
                self.drive(0.0, 0.0)
                room = self.room()
                self.say(f"Found a {d['label']} in the {self.nice(room or 'house')}!")
                self.found_it(t.get("fid"), q, d["bearing"])
                return True
        yaw = self.odom_pose[2]
        t["left"] -= abs(wrap(yaw - t["last"]))
        t["last"] = yaw
        if t["left"] <= 0:
            if t.get("final") and q != "*":
                self.say(t.get("give_up") or f"I couldn't find a {q}.")
                self.not_found(t.get("fid"))
            return True
        self.drive(0.0, self.scan_speed)
        return False

    def run_approach(self):
        """Face the object and drive up to it, stopping approach_distance away."""
        t = self.task
        d = self.visible(t["query"])
        now = self.now()
        if d:
            t["bearing"], t["seen_t"] = d["bearing"], now
            rng = self.det_range(d)
        else:
            rng = None
            if now - t["seen_t"] > 3.0:
                self.say(f"I'm next to the {t['query']}." if self.front_min < 1.2
                         else f"The {t['query']} is around here, but I lost sight of it on the way over.")
                return True
        x, y, _ = self.odom_pose
        if math.dist((x, y), t["start"]) > 6.0:
            self.say(f"I'm near the {t['query']}.")
            return True
        b = t["bearing"]
        if abs(b) > 0.1:
            w = clamp(1.5 * b, -0.6, 0.6)
            self.drive(0.0, w if abs(w) > 0.12 else math.copysign(0.12, w))
            t["bearing"] = b - w * 0.05            # dead-reckon between detections
            return False
        dist = min(v for v in (rng, self.front_min) if v is not None)
        if dist <= self.approach_dist:
            self.say(f"Here's the {t['query']}.")
            return True
        self.drive(clamp(0.6 * (dist - self.approach_dist), 0.06, 0.3), clamp(1.5 * b, -0.5, 0.5))
        return False

    def start_patrol(self):
        self.memory.items.pop("person", None)     # people move: count only those seen on this patrol
        rooms, order = self.rooms_by_distance(), []
        here = self.room()
        cur = here
        remaining = [r for r in rooms if r != here]
        while remaining:                         # nearest-neighbour tour
            nxt = min(remaining, key=lambda r: self.room_dist(cur, r))
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        self.say(f"Starting a patrol of {len(order) + 1} rooms: "
                 f"{', '.join(self.nice(r) for r in [here] + order if r)}.")
        steps = [{"type": "scan", "object": "*"}]
        for r in order:
            steps += [{"type": "goto", "place": r, "quiet": True},
                      {"type": "scan", "object": "*"}]
        steps.append({"type": "say", "text": "PATROL_SUMMARY"})
        self.queue.extendleft(reversed(steps))

    def room_dist(self, a, b):
        if a is None or a not in self.graph.nodes:
            return 0.0
        path = self.graph.shortest_path(a, b)
        return self.graph.path_length(path) if path else math.inf

    # --------------------------------------------------------- safety stop
    def blocked(self, forward):
        clear = (self.front_min if forward else self.rear_min) > self.stop_dist
        if clear:
            if self.blocked_since is not None:
                self.say("Path clear, continuing.")
            self.blocked_since = None
            return False
        if self.blocked_since is None:
            self.blocked_since = self.now()
            self.say("Something is in my way - waiting.")
        elif self.now() - self.blocked_since > self.blocked_timeout:
            self.say("Still blocked, giving up on this command.")
            self.queue.clear()
            self.task = {"kind": "abort"}
        self.drive(0.0, 0.0)
        return True

    def run_abort(self):
        return True

    # ------------------------------------------------------------ behaviours
    def run_wait(self):
        self.drive(0.0, 0.0)
        return self.now() >= self.task["until"]

    def run_move(self):
        t = self.task
        x, y, yaw = self.odom_pose
        sx, sy = t["start"]
        travelled = (x - sx) * math.cos(t["yaw0"]) + (y - sy) * math.sin(t["yaw0"])
        remaining = abs(t["dist"]) - abs(travelled)
        if remaining < 0.02:
            return True
        forward = t["dist"] >= 0
        if self.blocked(forward):
            return self.task.get("kind") == "abort"
        v = clamp(1.0 * remaining, 0.06, self.vmax) * (1 if forward else -1)
        w = clamp(2.0 * wrap(t["yaw0"] - yaw), -self.wmax, self.wmax)
        self.drive(v, w)
        return False

    def run_turn(self):
        t = self.task
        yaw = self.odom_pose[2]
        t["left"] -= wrap(yaw - t["last"])     # integrate so 360° turns work
        t["last"] = yaw
        if abs(t["left"]) < 0.02:
            return True
        w = clamp(2.0 * t["left"], -self.wmax, self.wmax)
        if abs(w) < 0.15:
            w = math.copysign(0.15, w)
        self.drive(0.0, w)
        return False

    def run_follow(self):
        t = self.task
        if self.scan is None:
            return False
        tr = t["tracker"]
        now = self.now()
        if self.scan_seq == t["seq"]:
            return False                       # wait for a new scan
        t["seq"] = self.scan_seq
        cands = self.person_candidates()
        if t["phase"] == "acquire":
            person = self.visible("person")
            prefer = person["bearing"] if person else None
            if tr.acquire(cands, self.odom_pose, prefer):
                tx, ty = fl.odom_to_base(self.odom_pose, tr.target)
                tr.last_seen = now
                t["phase"] = "track"
                self.say(f"I see you {math.hypot(tx, ty):.1f} m away - following. "
                         "Say 'stop' when you want me to stay.")
                return False
            yaw = self.odom_pose[2]
            t["spun"] += abs(wrap(yaw - t["last"]))
            t["last"] = yaw
            if t["spun"] > 2 * math.pi:
                self.say("I can't see anyone near me. Stand within 3 metres and say 'follow me' again.")
                return True
            self.drive(0.0, 0.6)
            return False
        target = tr.update(cands, self.odom_pose, now)
        if target is None:
            if tr.lost(now):
                self.say("I lost you - I'll wait here.")
                return True
            self.drive(0.0, 0.0)
            return False
        v, w = fl.follow_command(target, self.follow_dist, vmax=self.vmax + 0.1, wmax=self.wmax,
                                 front_clear=self.front_min if abs(math.atan2(target[1], target[0])) > 0.4
                                 else math.inf)
        self.drive(v, w)
        return False


def main():
    rclpy.init()
    node = Executor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, 0.0)
        node.save_memory()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

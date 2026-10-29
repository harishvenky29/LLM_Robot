"""Move the simulated person (the follow-me target).

  ros2 run llm_home_robot person_walker                 # walks kitchen -> living room -> hallway
  ros2 run llm_home_robot person_walker --ros-args -p route:="[bedroom, office]" -p loop:=true
  ros2 run llm_home_robot person_teleop                 # drive the person with the keyboard

  pub /person/cmd_vel  geometry_msgs/Twist   (VelocityControl plugin on the person model)
  sub /person/odom     nav_msgs/Odometry     (world-frame pose of the person)

The walker uses the same place graph as the robot, so the person walks
through doorways instead of walls.
"""
import math
import os
import select
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .place_graph import PlaceGraph


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PersonWalker(Node):
    def __init__(self):
        super().__init__("person_walker")
        share = get_package_share_directory("llm_home_robot")
        self.declare_parameter("places_file", os.path.join(share, "config", "places.yaml"))
        self.declare_parameter("route", ["kitchen", "living_room", "hallway"])
        self.declare_parameter("loop", False)
        self.declare_parameter("speed", 0.45)      # m/s, a slow walk
        self.declare_parameter("pause", 3.0)       # s at each stop
        self.declare_parameter("start_delay", 3.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.graph = PlaceGraph.from_files(gp("places_file"))
        self.route = [self.graph.resolve(r) or r for r in gp("route")]
        bad = [r for r in self.route if r not in self.graph.places]
        if bad:
            raise SystemExit(f"unknown places in route: {bad}. Known: {self.graph.place_names()}")
        self.loop, self.speed, self.pause = gp("loop"), gp("speed"), gp("pause")
        self.pose, self.pts, self.idx = None, [], 0
        self.wait_until = self.now() + gp("start_delay")
        self.pub = self.create_publisher(Twist, "/person/cmd_vel", 10)
        self.create_subscription(Odometry, "/person/odom", self.on_odom, 10)
        self.create_timer(0.05, self.step)
        self.get_logger().info(f"Person will walk: {' -> '.join(self.route)}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def drive(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)

    def step(self):
        if self.pose is None or self.now() < self.wait_until:
            if self.pose is not None:
                self.drive(0.0, 0.0)
            return
        x, y, yaw = self.pose
        if not self.pts:
            if self.idx >= len(self.route):
                if not self.loop:
                    self.drive(0.0, 0.0)
                    self.get_logger().info("Walk finished.")
                    raise SystemExit(0)
                self.idx = 0
            goal = self.route[self.idx]
            plan = self.graph.plan(x, y, goal)
            self.pts = list(plan[1]) if plan else []
            self.get_logger().info(f"Person walking to the {goal.replace('_', ' ')}")
            self.idx += 1
            return
        gx, gy = self.pts[0]
        dist = math.hypot(gx - x, gy - y)
        if dist < 0.2:
            self.pts.pop(0)
            if not self.pts:
                self.drive(0.0, 0.0)
                self.wait_until = self.now() + self.pause
            return
        err = wrap(math.atan2(gy - y, gx - x) - yaw)
        w = max(-1.2, min(1.2, 2.5 * err))
        v = self.speed * max(0.0, math.cos(err)) if abs(err) < 0.8 else 0.0
        self.drive(v, w)


KEYS = """
Walk the person:   w = forward   s = back   a = turn left   d = turn right
                   space = stop  q = quit
"""


class PersonTeleop(Node):
    def __init__(self):
        super().__init__("person_teleop")
        self.pub = self.create_publisher(Twist, "/person/cmd_vel", 10)

    def send(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)


def walker_main():
    rclpy.init()
    node = PersonWalker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.drive(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def teleop_main():
    import termios
    import tty
    rclpy.init()
    node = PersonTeleop()
    print(KEYS)
    old = termios.tcgetattr(sys.stdin)
    v = w = 0.0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = sys.stdin.read(1)
                if k == "w":
                    v, w = 0.5, 0.0
                elif k == "s":
                    v, w = -0.3, 0.0
                elif k == "a":
                    w = 0.8
                elif k == "d":
                    w = -0.8
                elif k == " ":
                    v = w = 0.0
                elif k == "q":
                    break
                print(f"\rspeed {v:+.1f} m/s  turn {w:+.1f} rad/s   ", end="", flush=True)
            node.send(v, w)
    finally:
        node.send(0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print()


if __name__ == "__main__":
    walker_main()

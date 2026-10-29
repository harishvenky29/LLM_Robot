"""RViz visualiser for the home robot.

  pub  /robot/markers   visualization_msgs/MarkerArray
       house (floors, walls, furniture), room + place labels, route graph,
       robot status label, planned route, speech bubble
  sub  /robot/status    std_msgs/String (JSON from the executor)
  sub  /robot/reply     std_msgs/String
  sub  /robot/objects   std_msgs/String (remembered objects -> labelled cubes)
  sub  /clicked_point   geometry_msgs/PointStamped   (RViz "Publish Point" tool)
  pub  /robot/command   std_msgs/String  "go to <nearest place>" on click

Learned places (~/.llm_home_robot/learned_places.yaml) appear in purple as
soon as the robot saves them.
"""
import json
import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from . import viz_markers as vm
from .place_graph import PlaceGraph

LEARNED_FILE = os.path.expanduser("~/.llm_home_robot/learned_places.yaml")

TYPES = {
    "cube": Marker.CUBE, "sphere": Marker.SPHERE, "cylinder": Marker.CYLINDER,
    "arrow": Marker.ARROW, "text": Marker.TEXT_VIEW_FACING,
    "line_strip": Marker.LINE_STRIP, "line_list": Marker.LINE_LIST,
    "sphere_list": Marker.SPHERE_LIST,
}


class Viz(Node):
    def __init__(self):
        super().__init__("robot_viz")
        share = get_package_share_directory("llm_home_robot")
        self.declare_parameter("places_file", os.path.join(share, "config", "places.yaml"))
        self.declare_parameter("learned_file", LEARNED_FILE)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("wall_height", 0.6)
        self.declare_parameter("show_graph", True)
        self.declare_parameter("show_aliases", True)
        self.declare_parameter("reply_seconds", 6.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.places_file, self.learned_file = gp("places_file"), gp("learned_file")
        self.frame = gp("frame_id")
        self.wall_h, self.show_graph = gp("wall_height"), gp("show_graph")
        self.show_aliases, self.reply_secs = gp("show_aliases"), gp("reply_seconds")

        self.graph, self.learned, self.learned_mtime = None, set(), None
        self.reload_places()
        self.status, self.reply, self.reply_t = None, None, 0.0
        self.last_goal = "__none__"

        self.pub = self.create_publisher(MarkerArray, "/robot/markers", 10)
        self.pub_cmd = self.create_publisher(String, "/robot/command", 10)
        self.create_subscription(String, "/robot/status", self.on_status, 10)
        self.create_subscription(String, "/robot/reply", self.on_reply, 10)
        self.create_subscription(PointStamped, "/clicked_point", self.on_click, 10)
        self.create_subscription(String, "/robot/objects", self.on_objects, 10)
        self.objects = []
        self.create_timer(2.0, self.publish_static)   # re-sent so late RViz windows get it
        self.create_timer(0.2, self.publish_dynamic)
        self.publish_static()
        self.get_logger().info("Publishing /robot/markers - add a MarkerArray display in RViz. "
                               "Use 'Publish Point' to click a place and send the robot there.")

    # ----------------------------------------------------------- inputs
    def reload_places(self):
        mtime = os.path.getmtime(self.learned_file) if os.path.exists(self.learned_file) else None
        if self.graph is not None and mtime == self.learned_mtime:
            return False
        base = PlaceGraph.from_files(self.places_file)
        self.graph = PlaceGraph.from_files(self.places_file, self.learned_file)
        self.learned = set(self.graph.places) - set(base.places)
        self.learned_mtime = mtime
        return True

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def on_objects(self, msg):
        try:
            self.objects = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.publish(vm.object_markers(self.objects))

    def on_reply(self, msg):
        self.reply, self.reply_t = msg.data, self.now()

    def on_click(self, msg):
        place = vm.nearest_place(self.graph, msg.point.x, msg.point.y)
        if place is None:
            self.get_logger().info(f"No known place near ({msg.point.x:.1f}, {msg.point.y:.1f}).")
            self.reply, self.reply_t = "No known place there - click closer to a blue pin.", self.now()
            return
        self.get_logger().info(f"Clicked {place} -> sending robot")
        self.pub_cmd.publish(String(data=f"go to {vm.nice(place)}"))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------- outputs
    def to_marker(self, d, stamp):
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = d["ns"], d["id"], TYPES[d["type"]], Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, (d["x"], d["y"], d["z"]))
        yaw = d.get("yaw", 0.0)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.scale.x, m.scale.y, m.scale.z = map(float, (d["sx"], d["sy"], d["sz"]))
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, d["rgba"])
        if "text" in d:
            m.text = d["text"]
        for px, py, pz in d.get("points", []):
            m.points.append(Point(x=float(px), y=float(py), z=float(pz)))
        m.lifetime = Duration(sec=0)
        return m

    def publish(self, dicts):
        stamp = self.get_clock().now().to_msg()
        self.pub.publish(MarkerArray(markers=[self.to_marker(d, stamp) for d in dicts]))

    def publish_static(self):
        if self.reload_places():
            # wipe old place markers (a learned place may have shifted the ids)
            self.pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            self.last_goal = "__none__"
        pins = [(q["x"], q["y"]) for q in self.graph.places.values()]
        dicts = vm.house_markers(self.graph.rooms, self.wall_h, pins)
        if self.show_graph:
            dicts += vm.graph_markers(self.graph)
        dicts += vm.place_markers(self.graph, self.goal(), self.learned, self.show_aliases)
        self.publish(dicts)

    def goal(self):
        return (self.status or {}).get("goal")

    def publish_dynamic(self):
        reply = self.reply if self.reply and self.now() - self.reply_t < self.reply_secs else None
        dicts = vm.robot_markers(self.status, reply)
        if self.goal() != self.last_goal:      # re-colour the goal pin straight away
            self.last_goal = self.goal()
            dicts += vm.place_markers(self.graph, self.goal(), self.learned, self.show_aliases)
        if dicts:
            self.publish(dicts)


def main():
    rclpy.init()
    node = Viz()
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

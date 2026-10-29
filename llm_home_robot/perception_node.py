"""Perception node: camera image -> object detections.

  sub  /camera/image         sensor_msgs/Image (rgb8 from Gazebo)
  sub  /camera/camera_info   sensor_msgs/CameraInfo (for the focal length)
  pub  /robot/detections     std_msgs/String  JSON {"stamp", "detections": [
                                 {"label", "conf", "bbox": [x1,y1,x2,y2], "bearing", "source"}]}
  pub  /camera/detections    sensor_msgs/Image  annotated image (view it in RViz)

Detectors (parameter `detector`):
  auto    - YOLO if ultralytics is installed, plus the colour detector (default)
  yolo    - YOLO only (COCO classes: person, chair, couch, tv, cup, bottle, ...)
  colour  - colour blobs only (red ball, blue box, green bin, yellow bucket); no downloads

The executor turns bearing + lidar range into a world position and remembers it.
No cv_bridge: images are converted with numpy so a pip-installed numpy/OpenCV
can't break ROS.
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from . import perception as pc

COLOURS_BGR = {"yolo": (40, 200, 255), "colour": (255, 120, 40)}


def image_to_rgb(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        img = img[:, :, :3]
        return img[:, :, ::-1].copy() if msg.encoding == "bgra8" else img.copy()
    raise ValueError(f"unsupported encoding {msg.encoding}")


def rgb_to_image(rgb, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding, msg.is_bigendian, msg.step = "rgb8", 0, rgb.shape[1] * 3
    msg.data = np.ascontiguousarray(rgb).tobytes()
    return msg


class Perception(Node):
    def __init__(self):
        super().__init__("perception")
        self.declare_parameter("detector", "auto")        # auto | yolo | colour
        self.declare_parameter("yolo_model", "yolo11n.pt")
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("rate", 4.0)               # detections per second
        self.declare_parameter("hfov", 1.2)               # used until camera_info arrives
        self.declare_parameter("publish_image", True)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        mode = gp("detector")
        self.hfov, self.fx, self.pub_img_on = gp("hfov"), None, gp("publish_image")
        self.use_colour = mode in ("auto", "colour", "both")
        self.yolo = None
        if mode in ("auto", "yolo", "both"):
            try:
                self.yolo = pc.YoloDetector(gp("yolo_model"), gp("yolo_conf"))
                self.get_logger().info(f"YOLO loaded ({gp('yolo_model')}).")
            except Exception as e:  # noqa: BLE001  (not installed / no internet for weights)
                msg = f"YOLO unavailable ({e.__class__.__name__}: {e})."
                if mode == "yolo":
                    raise RuntimeError(msg + " pip install ultralytics 'numpy<2'") from e
                self.get_logger().warn(msg + " Using the colour detector only. "
                                       "To enable: pip install ultralytics 'numpy<2' --break-system-packages")
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.use_colour = False
            self.get_logger().error("OpenCV missing: sudo apt install python3-opencv")
        self.latest = None
        self.pub = self.create_publisher(String, "/robot/detections", 10)
        self.pub_img = self.create_publisher(Image, "/camera/detections", 2)
        self.create_subscription(Image, "/camera/image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, qos_profile_sensor_data)
        self.create_timer(1.0 / gp("rate"), self.process)
        self.frames = 0
        self.get_logger().info(f"Perception ready: yolo={'on' if self.yolo else 'off'}, "
                               f"colour={'on' if self.use_colour else 'off'}")

    def on_info(self, msg):
        if msg.k[0] > 0:
            self.fx = msg.k[0]

    def on_image(self, msg):
        self.latest = msg

    def bearing(self, u, width):
        if self.fx:
            return math.atan2(width / 2.0 - u, self.fx)
        return pc.bearing_of(u, width, self.hfov)

    def process(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        try:
            rgb = image_to_rgb(msg)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=10.0)
            return
        dets = []
        if self.yolo is not None:
            dets += self.yolo(rgb)
        if self.use_colour:
            dets += pc.detect_colours(rgb)
        w = rgb.shape[1]
        for d in dets:
            x1, _y1, x2, _y2 = d["bbox"]
            d["bearing"] = round(self.bearing((x1 + x2) / 2.0, w), 3)
            d["bbox"] = [int(v) for v in d["bbox"]]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        fx = self.fx or (w / 2.0) / math.tan(self.hfov / 2.0)
        self.pub.publish(String(data=json.dumps({"stamp": stamp, "width": w, "height": rgb.shape[0],
                                                 "fx": round(fx, 1), "detections": dets})))
        self.frames += 1
        if self.frames == 1:
            self.get_logger().info(f"Camera OK ({w}x{rgb.shape[0]}).")
        if self.pub_img_on and self.pub_img.get_subscription_count() > 0:
            self.pub_img.publish(rgb_to_image(self.annotate(rgb, dets), msg.header))

    @staticmethod
    def annotate(rgb, dets):
        import cv2
        img = rgb[:, :, ::-1].copy()      # draw in BGR
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            c = COLOURS_BGR.get(d["source"], (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            text = f"{d['label']} {d['conf']:.2f}"
            cv2.putText(img, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        return img[:, :, ::-1]


def main():
    rclpy.init()
    node = Perception()
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

"""Terminal chat with the robot.

    ros2 run llm_home_robot chat

Type commands, see the robot's replies. Ctrl+C / 'quit' to exit.
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

BANNER = """
=== HomeBot chat ===  (type 'quit' to exit)
Move     go to the kitchen | move forward 2 meters | turn left 45 degrees | come back home
Look     find the red ball | is anyone in the kitchen | what do you see | where is the cup
People   follow me | stop             (walk the person: ros2 run llm_home_robot person_walker)
House    patrol the house | where are you | what places do you know
Learn    remember this place as reading corner
LLM      I'm thirsty | I need to do some work | someone is at the door
"""


class Chat(Node):
    def __init__(self):
        super().__init__("robot_chat")
        self.pub = self.create_publisher(String, "/robot/command", 10)
        self.create_subscription(String, "/robot/reply", self.on_reply, 10)

    def on_reply(self, msg):
        sys.stdout.write(f"\r\033[K🤖 {msg.data}\nyou> ")
        sys.stdout.flush()

    def send(self, text):
        self.pub.publish(String(data=text))


def main():
    rclpy.init()
    node = Chat()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(BANNER)
    try:
        while rclpy.ok():
            text = input("you> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                node.send(text)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

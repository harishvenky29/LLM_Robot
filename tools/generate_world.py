#!/usr/bin/env python3
"""Generate worlds/big_home.sdf and verify config/places.yaml against it.

Run from the package root:
    python3 tools/generate_world.py

It (1) writes a Gazebo Harmonic world with 7 rooms, doors, furniture,
coloured objects and a walkable person, (2) writes maps/big_home.{pgm,yaml}
(a ready-made Nav2 map of the same geometry) and config/props.yaml (Fuel
models the launch file spawns), and (3) checks that every place/node sits clear of walls and furniture and every
graph edge is a collision-free straight line for the robot. Edit the layout in
llm_home_robot/house_layout.py, then re-run - the check tells you if you broke a route.
"""
import math
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llm_home_robot.house_layout import (  # noqa: E402,F401  (re-exported for tools/offline_sim.py)
    FLOOR_RGB, FUEL, FURNITURE, OBJECTS, PERSON_START, PROPS, ROBOT_R, WALL_H, WALLS,
    object_box, obstacles, point_box_dist, segment_box_dist, wall_boxes, wall_segments)

MAP_RES = 0.05                      # m / pixel for the Nav2 map
MAP_ORIGIN = (-10.5, -7.5)          # bottom-left corner of the map (world frame)
MAP_SIZE = (21.0, 15.0)


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------
def validate(places_cfg):
    obs = obstacles()
    pts = {}
    for name, p in places_cfg["places"].items():
        pts[name] = (p["x"], p["y"])
    for name, p in places_cfg["nodes"].items():
        pts[name] = (p["x"], p["y"])
    errors = []
    for name, (x, y) in pts.items():
        for oname, *box in obs:
            d = point_box_dist(x, y, *box)
            if d < ROBOT_R:
                errors.append(f"point '{name}' ({x},{y}) only {d:.2f} m from {oname}")
    for a, b in places_cfg["edges"]:
        if a not in pts or b not in pts:
            errors.append(f"edge {a}-{b} references unknown node")
            continue
        for oname, *box in obs:
            d = segment_box_dist(*pts[a], *pts[b], box)
            if d < ROBOT_R:
                errors.append(f"edge {a}->{b} passes {d:.2f} m from {oname}")
    return errors


# ---------------------------------------------------------------------------
# SDF writer
# ---------------------------------------------------------------------------
def box_link(name, cx, cy, cz, sx, sy, sz, rgb, collide=True):
    r, g, b = rgb
    col = f"""
        <collision name="{name}_col">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>""" if collide else ""
    return f"""
      <link name="{name}">
        <pose>{cx:.3f} {cy:.3f} {cz:.3f} 0 0 0</pose>{col}
        <visual name="{name}_vis">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
          </material>
        </visual>
      </link>"""


GUI = """
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui><title>3D View</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.5 0.5 0.5</ambient_light>
        <background_color>0.85 0.88 0.92</background_color>
        <camera_pose>0 -14 24 0 1.05 1.5708</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="CameraTracking" name="Camera Tracking"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="MarkerManager" name="Marker manager"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="SelectEntities" name="Select Entities"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="VisualizationCapabilities" name="Visualization Capabilities"><gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui><title>World control</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">72</property><property type="double" key="width">121</property><property type="double" key="z">1</property><property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <play_pause>true</play_pause><step>true</step><start_paused>false</start_paused>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui><title>World stats</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">110</property><property type="double" key="width">290</property><property type="double" key="z">1</property><property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <sim_time>true</sim_time><real_time>true</real_time><real_time_factor>true</real_time_factor><iterations>true</iterations>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree"><gz-gui><property type="string" key="state">docked_collapsed</property></gz-gui></plugin>
      <plugin filename="ComponentInspector" name="Component inspector"><gz-gui><property type="string" key="state">docked_collapsed</property></gz-gui></plugin>
    </gui>"""


def object_link(obj):
    name, shape, cx, cy, size, h, (r, g, b) = obj
    if shape == "sphere":
        geom, cz = f"<sphere><radius>{size}</radius></sphere>", size
    elif shape == "box":
        geom, cz = f"<box><size>{size} {size} {h}</size></box>", h / 2
    else:
        geom, cz = f"<cylinder><radius>{size}</radius><length>{h}</length></cylinder>", h / 2
    return f"""
      <link name="{name}">
        <pose>{cx} {cy} {cz} 0 0 0</pose>
        <collision name="{name}_col"><geometry>{geom}</geometry></collision>
        <visual name="{name}_vis">
          <geometry>{geom}</geometry>
          <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
        </visual>
      </link>"""


def person_model():
    """A person you can drive around (/person/cmd_vel). Simple shapes for
    physics + lidar (legs at lidar height), Fuel mesh for looks. If the mesh
    can't be downloaded you just see the simple shapes."""
    x, y, yaw = PERSON_START
    mesh = FUEL + "Standing person/tip/files/meshes/standing.dae"
    leg = lambda side: f"""
        <collision name="leg_{side}">
          <pose>0 {0.1 if side == 'l' else -0.1} 0.47 0 0 0</pose>
          <geometry><cylinder><radius>0.07</radius><length>0.86</length></cylinder></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface>
        </collision>
        <visual name="leg_{side}_vis">
          <pose>0 {0.09 if side == 'l' else -0.09} 0.45 0 0 0</pose>
          <geometry><cylinder><radius>0.035</radius><length>0.8</length></cylinder></geometry>
          <material><ambient>0.15 0.2 0.35 1</ambient><diffuse>0.15 0.2 0.35 1</diffuse></material>
        </visual>"""
    return f"""
    <model name="person">
      <pose>{x} {y} 0.02 0 0 {yaw}</pose>
      <link name="body">
        <inertial>
          <pose>0 0 0.25 0 0 0</pose>
          <mass>70</mass>
          <inertia><ixx>40</ixx><iyy>40</iyy><izz>4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="feet">
          <pose>0 0 0.02 0 0 0</pose>
          <geometry><box><size>0.45 0.45 0.04</size></box></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface>
        </collision>{leg('l')}{leg('r')}
        <collision name="torso">
          <pose>0 0 1.25 0 0 0</pose>
          <geometry><box><size>0.25 0.45 0.7</size></box></geometry>
        </collision>
        <visual name="mesh">
          <pose>0 0 0.0 0.04 0 1.5708</pose>
          <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
        </visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_frame>person_odom</odom_frame>
        <robot_base_frame>person</robot_base_frame>
        <odom_topic>/person/odom</odom_topic>
        <tf_topic>/person/tf</tf_topic>
        <odom_publish_frequency>20</odom_publish_frequency>
        <dimensions>2</dimensions>
      </plugin>
    </model>"""


def build_sdf(rooms):
    walls = "".join(
        box_link(f"wall_{i}", cx, cy, WALL_H / 2, sx, sy, WALL_H, (0.93, 0.91, 0.87))
        for i, (cx, cy, sx, sy) in enumerate(wall_boxes()))
    floors = "".join(
        box_link(f"floor_{name}", (x0 + x1) / 2, (y0 + y1) / 2, 0.002,
                 x1 - x0, y1 - y0, 0.004, FLOOR_RGB.get(name, (0.7, 0.7, 0.7)), collide=False)
        for name, (x0, y0, x1, y1) in rooms.items())
    furniture = "".join(
        box_link(n, cx, cy, h / 2, sx, sy, h, rgb) for n, cx, cy, sx, sy, h, rgb in FURNITURE)
    objects = "".join(object_link(o) for o in OBJECTS)

    return f"""<?xml version="1.0" ?>
<!-- GENERATED by tools/generate_world.py - edit that file, not this one. -->
<sdf version="1.9">
  <world name="big_home">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
{GUI}
    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.85 0.88 0.92 1</background>
      <shadows>false</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material><ambient>0.55 0.6 0.5 1</ambient><diffuse>0.55 0.6 0.5 1</diffuse></material>
        </visual>
      </link>
    </model>

    <model name="house">
      <static>true</static>{floors}{walls}
    </model>

    <model name="furniture">
      <static>true</static>{furniture}
    </model>

    <model name="objects">
      <static>true</static>{objects}
    </model>
{person_model()}
  </world>
</sdf>
"""


def write_map(path_base):
    """Occupancy grid of walls/furniture/objects/props for Nav2 (PGM + YAML)."""
    w, h = int(round(MAP_SIZE[0] / MAP_RES)), int(round(MAP_SIZE[1] / MAP_RES))
    ox, oy = MAP_ORIGIN
    grid = bytearray([205]) * (w * h)                 # unknown outside the house
    hx0, hy0, hx1, hy1 = -10.0, -7.0, 10.0, 7.0
    for row in range(h):
        y = oy + (h - row - 0.5) * MAP_RES
        for col in range(w):
            x = ox + (col + 0.5) * MAP_RES
            if hx0 <= x <= hx1 and hy0 <= y <= hy1:
                grid[row * w + col] = 254
    for _n, cx, cy, sx, sy in obstacles():
        c0 = max(int((cx - sx / 2 - ox) / MAP_RES), 0)
        c1 = min(int(math.ceil((cx + sx / 2 - ox) / MAP_RES)), w)
        r0 = max(int(h - math.ceil((cy + sy / 2 - oy) / MAP_RES)), 0)
        r1 = min(int(h - (cy - sy / 2 - oy) / MAP_RES), h)
        for row in range(r0, r1):
            grid[row * w + c0: row * w + c1] = bytes(c1 - c0)
    with open(path_base + ".pgm", "wb") as f:
        f.write(f"P5\n# llm_home_robot generated map\n{w} {h}\n255\n".encode())
        f.write(bytes(grid))
    with open(path_base + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(path_base)}.pgm\nmode: trinary\nresolution: {MAP_RES}\n"
                f"origin: [{ox}, {oy}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return w, h


def write_props(path):
    props = [{"name": n, "uri": FUEL + m, "x": x, "y": y, "z": z, "yaw": yaw}
             for n, m, x, y, z, yaw, _fp in PROPS]
    with open(path, "w") as f:
        f.write("# GENERATED by tools/generate_world.py - Fuel models spawned by home.launch.py\n")
        yaml.safe_dump({"props": props}, f, sort_keys=False)


def main():
    with open(os.path.join(ROOT, "config", "places.yaml")) as f:
        cfg = yaml.safe_load(f)
    errors = validate(cfg)
    out = os.path.join(ROOT, "worlds", "big_home.sdf")
    with open(out, "w") as f:
        f.write(build_sdf(cfg["rooms"]))
    print(f"wrote {out}  ({len(wall_boxes())} wall pieces, {len(FURNITURE)} furniture items, "
          f"{len(OBJECTS)} coloured objects, 1 person)")
    os.makedirs(os.path.join(ROOT, "maps"), exist_ok=True)
    w, h = write_map(os.path.join(ROOT, "maps", "big_home"))
    print(f"wrote maps/big_home.pgm ({w}x{h} px @ {MAP_RES} m)")
    write_props(os.path.join(ROOT, "config", "props.yaml"))
    print(f"wrote config/props.yaml ({len(PROPS)} Fuel props)")
    if errors:
        print("\nLAYOUT PROBLEMS:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    n = len(cfg["places"]) + len(cfg["nodes"])
    print(f"layout OK: {n} points and {len(cfg['edges'])} routes all clear by >= {ROBOT_R} m")


if __name__ == "__main__":
    main()

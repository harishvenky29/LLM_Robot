"""Builds RViz marker descriptions for the house, places and robot state.

Pure Python: returns plain dicts so the logic is testable without ROS.
viz_node.py turns each dict into a visualization_msgs/Marker.

Dict keys: ns, id, type (cube|sphere|cylinder|text|line_strip|line_list|arrow),
           x, y, z, sx, sy, sz, rgba, and optionally text, points [(x, y, z)].
"""
import math

from . import house_layout as house

# palette
WALL_RGBA = (0.93, 0.91, 0.87, 1.0)
PLACE_RGBA = (0.11, 0.49, 0.84, 1.0)        # blue
GOAL_RGBA = (0.18, 0.75, 0.35, 1.0)         # green
LEARNED_RGBA = (0.62, 0.30, 0.85, 1.0)      # purple
ROUTE_RGBA = (0.91, 0.35, 0.05, 1.0)        # orange, matches the robot
GRAPH_RGBA = (0.45, 0.45, 0.45, 0.55)
ROOM_TEXT_RGBA = (0.15, 0.15, 0.15, 1.0)
PLACE_TEXT_RGBA = (0.97, 0.97, 0.92, 1.0)   # near-white, readable on every floor
GOAL_TEXT_RGBA = (0.35, 1.00, 0.50, 1.0)

STATE_RGBA = {
    "idle": (0.18, 0.75, 0.35, 1.0),
    "busy": (0.91, 0.35, 0.05, 1.0),
    "blocked": (0.85, 0.10, 0.10, 1.0),
}


def nice(name):
    return name.replace("_", " ")


def _m(ns, i, kind, x=0.0, y=0.0, z=0.0, sx=1.0, sy=1.0, sz=1.0, rgba=(1, 1, 1, 1), **kw):
    d = {"ns": ns, "id": i, "type": kind, "x": x, "y": y, "z": z,
         "sx": sx, "sy": sy, "sz": sz, "rgba": tuple(rgba)}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# Static scene: floors, walls, furniture, room names
# ---------------------------------------------------------------------------
def _room_label_spot(name, bounds, places=()):
    """Pick a spot for the room name that doesn't sit on furniture or a place
    label: try along the far wall, then the near wall, centre first."""
    x0, y0, x1, y1 = bounds
    w, h = 0.33 * len(name) + 0.4, 0.8          # rough text footprint (m)
    cx = (x0 + x1) / 2
    if name == "hallway":
        rows = [(y0 + y1) / 2 + 0.45]
    elif y0 >= 0:
        rows = [y1 - 0.6, y0 + 0.75]
    else:
        rows = [y0 + 0.6, y1 - 0.75]
    xs = [cx + k * d for d in (0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0) for k in ((1,) if d == 0 else (-1, 1))]
    blockers = [(fx, fy, fsx + w, fsy + h) for _n, fx, fy, fsx, fsy, _h, _c in house.FURNITURE]
    blockers += [(px, py + 0.3, 2.2 + w, 0.9 + h) for px, py in places]
    for ty in rows:
        for tx in xs:
            if tx - w / 2 < x0 or tx + w / 2 > x1:
                continue
            if all(abs(tx - bx) > bsx / 2 or abs(ty - by) > bsy / 2 for bx, by, bsx, bsy in blockers):
                return tx, ty
    return cx, rows[0]


def house_markers(rooms, wall_height=0.6, places=()):
    """Walls are drawn shorter than in Gazebo (default 0.6 m) so the top-down
    view stays readable; set wall_height=house.WALL_H for full height.
    places: [(x, y), ...] of place pins, so room names avoid their labels."""
    out = []
    for i, (name, (x0, y0, x1, y1)) in enumerate(sorted(rooms.items())):
        r, g, b = house.FLOOR_RGB.get(name, (0.7, 0.7, 0.7))
        out.append(_m("floors", i, "cube", (x0 + x1) / 2, (y0 + y1) / 2, -0.01,
                      x1 - x0, y1 - y0, 0.01, (r, g, b, 0.55)))
        tx, ty = _room_label_spot(name, (x0, y0, x1, y1), places)
        out.append(_m("room_names", i, "text", tx, ty, 0.9,
                      sz=0.55, rgba=ROOM_TEXT_RGBA, text=nice(name).upper()))
    for i, (cx, cy, sx, sy) in enumerate(house.wall_boxes()):
        out.append(_m("walls", i, "cube", cx, cy, wall_height / 2, sx, sy, wall_height, WALL_RGBA))
    for i, (name, cx, cy, sx, sy, h, (r, g, b)) in enumerate(house.FURNITURE):
        out.append(_m("furniture", i, "cube", cx, cy, h / 2, sx, sy, h, (r, g, b, 0.9)))
        out.append(_m("furniture_names", i, "text", cx, cy, h + 0.15,
                      sz=0.18, rgba=(0.3, 0.3, 0.3, 0.9), text=nice(name)))
    return out


# ---------------------------------------------------------------------------
# Knowledge: places, aliases, route graph
# ---------------------------------------------------------------------------
def place_markers(graph, goal=None, learned=(), show_aliases=True):
    out = []
    for i, name in enumerate(sorted(graph.places)):
        p = graph.places[name]
        rgba = GOAL_RGBA if name == goal else (LEARNED_RGBA if name in learned else PLACE_RGBA)
        big = name == goal
        out.append(_m("place_pins", i, "cylinder", p["x"], p["y"], 0.03,
                      0.3 if big else 0.2, 0.3 if big else 0.2, 0.06, rgba))
        label = nice(name)
        aliases = [a for a in p["aliases"] if a.lower() != label]
        if show_aliases and aliases:
            label += "\n(" + ", ".join(aliases[:2]) + ")"
        crowded = any(o != name and abs(q["x"] - p["x"]) < 1.6 and 0 < q["y"] - p["y"] < 1.2
                      for o, q in graph.places.items())
        ly = p["y"] - 0.5 if crowded else p["y"] + 0.35
        out.append(_m("place_names", i, "text", p["x"], ly, 0.3,
                      sz=0.26 if big else 0.22,
                      rgba=GOAL_TEXT_RGBA if big else PLACE_TEXT_RGBA, text=label))
        if p["yaw"] is not None:   # preferred facing direction at arrival
            out.append(_m("place_headings", i, "arrow", p["x"], p["y"], 0.08,
                          0.35, 0.05, 0.05, rgba[:3] + (0.8,), yaw=p["yaw"]))
    return out


def graph_markers(graph):
    pts, seen = [], set()
    for a, nbrs in graph.adj.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            (ax, ay, _), (bx, by, _) = graph.nodes[a], graph.nodes[b]
            pts += [(ax, ay, 0.02), (bx, by, 0.02)]
    waypoints = [n for n in graph.nodes if n not in graph.places]
    out = [_m("route_graph", 0, "line_list", sx=0.025, rgba=GRAPH_RGBA, points=pts)]
    out.append(_m("waypoints", 0, "sphere_list", sx=0.09, sy=0.09, sz=0.09,
                  rgba=(0.45, 0.45, 0.45, 0.8),
                  points=[(graph.nodes[n][0], graph.nodes[n][1], 0.03) for n in waypoints]))
    return out


# ---------------------------------------------------------------------------
# Dynamic: robot status, planned route, speech bubble
# ---------------------------------------------------------------------------
def robot_markers(status, reply=None):
    """status: dict from /robot/status. reply: latest /robot/reply text or None."""
    if not status:
        return []
    x, y = status["x"], status["y"]
    state = status.get("state", "idle")
    rgba = STATE_RGBA.get(state, STATE_RGBA["idle"])
    out = []

    head = f"HomeBot | {state}"
    if status.get("following"):
        head += " | following you"
    elif status.get("looking_for"):
        head += f" | looking for {status['looking_for']}"
    if status.get("goal"):
        head += f" -> {nice(status['goal'])}"
    room = status.get("room")
    head += f"\n{nice(room) if room else 'unknown'} ({x:.1f}, {y:.1f})"
    out.append(_m("robot_status", 0, "text", x, y, 0.75, sz=0.24, rgba=rgba, text=head))
    out.append(_m("robot_ring", 0, "cylinder", x, y, 0.01, 0.75, 0.75, 0.015,
                  rgba[:3] + (0.35,)))

    route = status.get("route") or []
    pts = [(x, y, 0.06)] + [(px, py, 0.06) for px, py in route]
    # always publish (possibly 2 identical points) so an old route gets overwritten
    if len(pts) < 2:
        pts.append(pts[0])
    out.append(_m("planned_route", 0, "line_strip", sx=0.07,
                  rgba=ROUTE_RGBA if route else (0, 0, 0, 0), points=pts))

    person = status.get("person")
    px, py = person if person else (x, y)
    out.append(_m("person", 0, "cylinder", px, py, 0.9, 0.5, 0.5, 1.8,
                  PERSON_RGBA[:3] + (0.35,) if person else (0, 0, 0, 0)))
    out.append(_m("person", 1, "text", px, py, 2.0, sz=0.25,
                  rgba=PERSON_RGBA if person else (0, 0, 0, 0), text="YOU" if person else " "))
    sees = status.get("sees") or []
    out.append(_m("search", 0, "text", x, y, 0.45, sz=0.18,
                  rgba=(0.9, 0.9, 0.5, 1.0) if sees else (0, 0, 0, 0),
                  text=("sees: " + ", ".join(sees[:4])) if sees else " "))
    out.append(_m("speech", 0, "text", x, y, 1.25, sz=0.22,
                  rgba=(1.0, 1.0, 1.0, 1.0) if reply else (0, 0, 0, 0),
                  text=_wrap(f'"{reply}"', 38) if reply else " "))
    return out


MAX_OBJECTS = 40
OBJECT_RGBA = {"red": (0.9, 0.1, 0.1, 0.95), "blue": (0.1, 0.3, 0.95, 0.95),
               "green": (0.1, 0.75, 0.15, 0.95), "yellow": (0.95, 0.85, 0.1, 0.95),
               "person": (1.0, 0.55, 0.75, 0.95)}
DEFAULT_OBJECT_RGBA = (0.95, 0.95, 0.95, 0.9)
PERSON_RGBA = (0.2, 0.9, 0.9, 1.0)


def object_markers(objects, now=None):
    """Remembered objects (semantic map): a small cube + label each. Always
    returns MAX_OBJECTS slots so forgotten objects get hidden."""
    import time as _t
    now = _t.time() if now is None else now
    out = []
    objs = sorted(objects or [], key=lambda o: -o.get("seen", 0))[:MAX_OBJECTS]
    for i in range(MAX_OBJECTS):
        if i < len(objs):
            o = objs[i]
            key = o["label"].split()[0]
            rgba = OBJECT_RGBA.get(key, OBJECT_RGBA.get(o["label"], DEFAULT_OBJECT_RGBA))
            age = now - o.get("seen", now)
            if age > 600:                        # fade things not seen for 10 min
                rgba = rgba[:3] + (0.4,)
            out.append(_m("objects", i, "cube", o["x"], o["y"], 0.12, 0.22, 0.22, 0.22, rgba))
            out.append(_m("object_names", i, "text", o["x"], o["y"], 0.5, sz=0.2,
                          rgba=rgba[:3] + (1.0,), text=_wrap(o["label"], 30)))
        else:
            out.append(_m("objects", i, "cube", rgba=(0, 0, 0, 0), sx=0.01, sy=0.01, sz=0.01))
            out.append(_m("object_names", i, "text", sz=0.01, rgba=(0, 0, 0, 0), text=" "))
    return out


def _wrap(text, width):
    # RViz's text font is ASCII-only
    text = text.replace("°", " deg").encode("ascii", "ignore").decode()
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:4])


def nearest_place(graph, x, y, max_dist=1.5):
    """Place closest to a clicked point, or None if nothing within max_dist."""
    if not graph.places:
        return None
    name = min(graph.places, key=lambda n: math.dist((x, y), graph.nodes[n][:2]))
    return name if math.dist((x, y), graph.nodes[name][:2]) <= max_dist else None

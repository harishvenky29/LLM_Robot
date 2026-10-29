"""Single source of truth for the house geometry.

Used by tools/generate_world.py (to write the Gazebo SDF), the RViz marker
node (to draw the same walls and furniture) and the offline simulator.
Pure Python, no ROS.
"""
import math

WALL_T = 0.15        # wall thickness (m)
WALL_H = 2.5         # wall height (m)
DOOR_W = 1.2         # standard door width (m)
ROBOT_R = 0.28       # robot footprint radius incl. margin (m)

# --- Walls: (x1, y1, x2, y2, [(door_centre, door_width), ...]) -------------
WALLS = [
    # outer shell 20 m x 14 m
    (-10, 7, 10, 7, []),
    (-10, -7, 10, -7, []),
    (-10, -7, -10, 7, []),
    (10, -7, 10, 7, []),
    # hallway walls with doors into each room
    (-10, 1, 10, 1, [(-6, DOOR_W), (1, DOOR_W), (7, DOOR_W)]),
    (-10, -1, 10, -1, [(-6, DOOR_W), (-1, DOOR_W), (5.5, DOOR_W)]),
    # north partitions
    (-2, 1, -2, 7, [(5, DOOR_W)]),          # living | kitchen (door)
    (4, 1, 4, 7, [(4, 2.0)]),               # kitchen | dining (open arch)
    # south partitions
    (-3, -7, -3, -1, []),                   # bedroom | bathroom
    (1, -7, 1, -1, []),                     # bathroom | office
]

# --- Furniture: name, cx, cy, sx, sy, height, rgb ----------------------------
FURNITURE = [
    ("sofa",          -6.0,  6.3, 2.2, 0.9, 0.8, (0.35, 0.30, 0.55)),
    ("coffee_table",  -6.0,  5.2, 1.0, 0.6, 0.45, (0.55, 0.40, 0.25)),
    ("tv_stand",      -9.6,  4.0, 0.5, 1.8, 0.6, (0.15, 0.15, 0.15)),
    ("counter",        0.9,  6.5, 3.6, 0.6, 0.9, (0.85, 0.85, 0.80)),
    ("island",         1.5,  5.2, 1.6, 0.8, 0.9, (0.75, 0.75, 0.70)),
    ("fridge",         3.4,  6.4, 0.7, 0.7, 1.8, (0.90, 0.92, 0.95)),
    ("dining_table",   7.5,  5.2, 2.0, 1.0, 0.75, (0.50, 0.33, 0.20)),
    ("sideboard",      9.6,  3.0, 0.5, 1.6, 0.9, (0.40, 0.27, 0.16)),
    ("bed",           -7.0, -5.9, 1.6, 2.0, 0.55, (0.70, 0.75, 0.90)),
    ("wardrobe",      -9.6, -3.0, 0.6, 1.6, 2.0, (0.45, 0.32, 0.22)),
    ("bathtub",       -1.0, -6.2, 1.7, 0.8, 0.6, (0.95, 0.95, 0.98)),
    ("sink",           0.6, -2.5, 0.5, 0.5, 0.85, (0.95, 0.95, 0.98)),
    ("desk",           5.5, -6.3, 1.6, 0.8, 0.75, (0.60, 0.45, 0.30)),
    ("bookshelf",      9.7, -4.0, 0.4, 2.0, 1.8, (0.42, 0.30, 0.20)),
]

# --- Brightly coloured objects the camera can always find ---------------------
# name, shape (sphere|box|cylinder), cx, cy, size (radius or edge), height, rgb
# Saturated colours on purpose: the colour detector ignores the muted
# furniture/floor colours, so these are found even without YOLO.
OBJECTS = [
    ("red_ball",      "sphere",   -8.3,  2.8, 0.15, 0.30, (0.90, 0.05, 0.05)),
    ("blue_box",      "box",      -8.6, -1.8, 0.35, 0.35, (0.05, 0.25, 0.95)),
    ("green_bin",     "cylinder",  2.2, -6.3, 0.18, 0.45, (0.05, 0.75, 0.10)),
    ("yellow_bucket", "cylinder",  9.2,  6.2, 0.16, 0.30, (0.95, 0.85, 0.05)),
]

# --- Realistic props from Gazebo Fuel (downloaded on first launch) ------------
# These give YOLO real-looking things to recognise (person, chair, toilet, cup).
# They are spawned after the world starts, so an offline machine still runs.
# name, fuel model, x, y, z, yaw, footprint (sx, sy) or None if on furniture
FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"
PROPS = [
    ("visitor",      "Standing person", -9.1, -5.6, 0.0,   1.57, (0.6, 0.6)),
    ("office_guest", "Casual female",    8.6, -2.3, 0.0,   3.14, (0.7, 0.7)),
    ("dining_chair", "Chair",            6.0,  5.2, 0.0,   0.0,  (0.65, 0.65)),
    ("desk_chair",   "Chair",            8.0, -5.8, 0.0,   1.57, (0.65, 0.65)),
    ("toilet",       "Toilet",          -2.45, -5.0, 0.0,  0.0,  (0.8, 0.65)),
    ("kitchen_cup",  "Plastic Cup",      1.2,  5.1, 0.97,  0.0,  None),
    ("coke",         "Coke",             7.9,  5.3, 0.75,  0.0,  None),
]

# --- The person you can walk around (follow-me target) -----------------------
PERSON_START = (-4.5, 2.2, 0.0)     # living room, x, y, yaw
PERSON_RADIUS = 0.25

# --- Floor colours per room ---------------------------------------------------
FLOOR_RGB = {
    "living_room": (0.80, 0.72, 0.60),
    "kitchen":     (0.85, 0.85, 0.85),
    "dining_room": (0.78, 0.68, 0.55),
    "hallway":     (0.65, 0.60, 0.55),
    "bedroom":     (0.70, 0.75, 0.80),
    "bathroom":    (0.80, 0.90, 0.92),
    "office":      (0.72, 0.70, 0.65),
}


def wall_segments():
    """Split walls around door gaps -> (x1, y1, x2, y2) pieces."""
    for x1, y1, x2, y2, doors in WALLS:
        horizontal = (y1 == y2)
        a, b = (x1, x2) if horizontal else (y1, y2)
        cuts = sorted((c - w / 2, c + w / 2) for c, w in doors)
        pieces, cur = [], a
        for lo, hi in cuts:
            if lo > cur:
                pieces.append((cur, lo))
            cur = hi
        if cur < b:
            pieces.append((cur, b))
        for s0, s1 in pieces:
            yield (s0, y1, s1, y1) if horizontal else (x1, s0, x1, s1)


def wall_boxes():
    """Walls as axis-aligned boxes (cx, cy, sx, sy)."""
    boxes = []
    for x1, y1, x2, y2 in wall_segments():
        if y1 == y2:  # horizontal: extend by half thickness so corners close
            boxes.append(((x1 + x2) / 2, y1, abs(x2 - x1) + WALL_T, WALL_T))
        else:
            boxes.append((x1, (y1 + y2) / 2, WALL_T, abs(y2 - y1) + WALL_T))
    return boxes


def object_box(obj):
    """Footprint (cx, cy, sx, sy) of a coloured object."""
    _n, shape, cx, cy, size, _h, _c = obj
    d = size if shape == "box" else 2 * size
    return cx, cy, d, d


def obstacles(objects=True, props=True):
    """[(name, cx, cy, sx, sy), ...] for walls, furniture, objects and props."""
    obs = [("wall", *b) for b in wall_boxes()]
    obs += [(n, cx, cy, sx, sy) for n, cx, cy, sx, sy, _h, _c in FURNITURE]
    if objects:
        obs += [(o[0], *object_box(o)) for o in OBJECTS]
    if props:
        obs += [(n, x, y, fp[0], fp[1]) for n, _m, x, y, _z, _yaw, fp in PROPS if fp]
    return obs


def point_box_dist(px, py, cx, cy, sx, sy):
    dx = max(abs(px - cx) - sx / 2, 0.0)
    dy = max(abs(py - cy) - sy / 2, 0.0)
    return math.hypot(dx, dy)


def segment_box_dist(ax, ay, bx, by, box, step=0.02):
    n = max(int(math.hypot(bx - ax, by - ay) / step), 1)
    return min(point_box_dist(ax + (bx - ax) * i / n, ay + (by - ay) * i / n, *box)
               for i in range(n + 1))

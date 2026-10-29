"""Vision helpers: detectors, camera geometry, object names and object memory.

Pure Python + numpy (+ OpenCV for the colour detector), no ROS, so it is
unit-testable. perception_node.py wraps the detectors; the executor uses the
geometry, matching and memory.
"""
import math
import re
import time

# ---------------------------------------------------------------------------
# Object vocabulary
# ---------------------------------------------------------------------------
# what people say -> what the detectors call it (COCO names for YOLO)
SYNONYMS = {
    "person": "person", "people": "person", "someone": "person", "somebody": "person",
    "anyone": "person", "anybody": "person", "human": "person", "man": "person",
    "woman": "person", "guest": "person", "visitor": "person",
    "sofa": "couch", "couch": "couch", "settee": "couch",
    "fridge": "refrigerator", "refrigerator": "refrigerator",
    "tv": "tv", "television": "tv", "telly": "tv",
    "table": "dining table", "dining table": "dining table",
    "plant": "potted plant", "potted plant": "potted plant",
    "mug": "cup", "cup": "cup", "glass": "cup",
    "can": "bottle", "coke": "bottle", "drink": "bottle", "bottle": "bottle",
    "ball": "ball", "sports ball": "ball",
    "chair": "chair", "seat": "chair", "toilet": "toilet", "loo": "toilet", "bed": "bed",
    "sink": "sink", "laptop": "laptop", "book": "book", "clock": "clock",
    "keys": "keys", "phone": "cell phone", "mobile": "cell phone", "cell phone": "cell phone",
    "remote": "remote", "backpack": "backpack", "bag": "handbag", "umbrella": "umbrella",
    "box": "box", "bin": "bin", "bucket": "bucket",
}
COLOURS = ("red", "green", "blue", "yellow")
# coloured objects in the house (see house_layout.OBJECTS) -> canonical labels
COLOUR_SHAPE = {"red": "ball", "blue": "box", "green": "bin", "yellow": "bucket"}

# where things usually are, best guess first (used to order the search)
ROOM_HINTS = {
    "cup": ["kitchen", "dining_room", "office", "living_room"],
    "bottle": ["kitchen", "dining_room", "living_room"],
    "refrigerator": ["kitchen"],
    "person": ["living_room", "kitchen", "office", "bedroom", "dining_room"],
    "couch": ["living_room"], "tv": ["living_room", "bedroom"],
    "bed": ["bedroom"], "toilet": ["bathroom"], "sink": ["bathroom", "kitchen"],
    "laptop": ["office", "living_room"], "book": ["office", "living_room", "bedroom"],
    "chair": ["dining_room", "office", "kitchen"], "dining table": ["dining_room", "kitchen"],
    "keys": ["hallway", "kitchen", "living_room"], "cell phone": ["living_room", "bedroom", "office"],
    "ball": ["living_room", "bedroom"], "box": ["bedroom", "office"],
}


# what the robot bothers to remember (YOLO knows 80 COCO classes; in a house
# "airplane" or "train" are always false alarms)
HOME_LABELS = {
    "person", "chair", "couch", "bed", "toilet", "tv", "sink", "refrigerator", "cup", "bottle",
    "wine glass", "bowl", "laptop", "book", "potted plant", "dining table", "clock", "vase",
    "cell phone", "remote", "keyboard", "mouse", "microwave", "oven", "backpack", "handbag",
    "umbrella", "teddy bear", "ball", "cat", "dog",
}


def rememberable(label):
    return label in HOME_LABELS or label.split()[0] in COLOURS


def canonical_object(text):
    """'my red ball' -> 'red ball', 'someone' -> 'person', 'the sofa' -> 'couch'."""
    t = re.sub(r"[^a-z ]", " ", text.lower())
    t = re.sub(r"\b(the|a|an|my|our|some|any|your|his|her|their)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    words = t.split()
    colour = next((w for w in words if w in COLOURS), None)
    rest = " ".join(w for w in words if w not in COLOURS)
    base = None
    for k in sorted(SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", rest):
            base = SYNONYMS[k]
            break
    if colour:
        return f"{colour} {base or COLOUR_SHAPE[colour]}"
    if base:
        return base
    return rest if len(rest.split()) <= 2 else None


def matches(query, label):
    """Does a detector label satisfy the request? ('ball' ~ 'sports ball',
    'red ball' ~ 'red ball', 'bottle' ~ 'cup' no)."""
    q, lab = canonical_object(query) or query, canonical_object(label) or label
    if q == lab:
        return True
    qw, lw = q.split(), lab.split()
    if qw[0] in COLOURS:                 # colour must match if asked for
        return lw[0] == qw[0] and (len(qw) == 1 or qw[-1] == lw[-1] or qw[-1] in COLOUR_SHAPE.values())
    return qw[-1] == lw[-1]              # 'ball' matches 'red ball' / 'sports ball'


def search_rooms(query, rooms_by_distance):
    """Room visiting order: hinted rooms first, then the rest nearest-first."""
    base = (canonical_object(query) or query).split()[-1]
    key = canonical_object(query) or query
    hints = ROOM_HINTS.get(key) or ROOM_HINTS.get(base) or []
    order = [r for r in hints if r in rooms_by_distance]
    return order + [r for r in rooms_by_distance if r not in order]


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------
def bearing_of(u, width, hfov):
    """Horizontal image coordinate -> bearing (rad, + = left) for a pinhole camera."""
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return math.atan2((width / 2.0) - u, fx)


def range_at_bearing(ranges, angle_min, angle_inc, bearing, half_width=0.05, rmin=0.1, rmax=12.0):
    """Median lidar range within +-half_width of a bearing (None if nothing)."""
    vals = []
    n = len(ranges)
    if n == 0:
        return None
    for k in range(-int(half_width / angle_inc) - 1, int(half_width / angle_inc) + 2):
        a = bearing + k * angle_inc
        i = int(round((math.atan2(math.sin(a), math.cos(a)) - angle_min) / angle_inc)) % n
        r = ranges[i]
        if rmin < r < rmax and math.isfinite(r):
            vals.append(r)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def to_world(pose, bearing, rng, sensor_offset=0.05):
    x, y, yaw = pose
    sx, sy = x + sensor_offset * math.cos(yaw), y + sensor_offset * math.sin(yaw)
    return sx + rng * math.cos(yaw + bearing), sy + rng * math.sin(yaw + bearing)


# ---------------------------------------------------------------------------
# Colour detector (OpenCV) - always works in the sim, no model download
# ---------------------------------------------------------------------------
# OpenCV HSV: H 0-180, S/V 0-255. High saturation threshold keeps the muted
# furniture, floors and walls out.
HSV_RANGES = {
    "red":    [((0, 170, 80), (8, 255, 255)), ((172, 170, 80), (180, 255, 255))],
    "yellow": [((20, 170, 90), (34, 255, 255))],
    "green":  [((45, 170, 60), (80, 255, 255))],
    # hue 106+ and V 90+ keeps out the dark teal upholstery of the Fuel chair (H~100, V~60)
    "blue":   [((106, 170, 90), (125, 255, 255))],
}
# the red ball must look round (the Coke can is red too, but tall)
SHAPE_CHECK = {"red": (0.7, 1.45, 0.6)}      # min aspect h/w, max aspect, min circularity

# typical real heights (m) - lets the executor sanity-check the lidar range
OBJECT_HEIGHTS = {
    "person": 1.7, "chair": 0.85, "couch": 0.9, "tv": 0.6, "bed": 0.6, "toilet": 0.8,
    "refrigerator": 1.75, "dining table": 0.75, "cup": 0.12, "bottle": 0.22, "laptop": 0.25,
    "potted plant": 0.6, "sink": 0.3, "book": 0.22,
    "red ball": 0.30, "blue box": 0.35, "green bin": 0.45, "yellow bucket": 0.30,
}


def estimate_range(label, bbox, img_h, fx, cam_height=0.6):
    """Distance from apparent size (pinhole). None if unknown or the box is cut
    off by the image edge."""
    h_real = OBJECT_HEIGHTS.get(label)
    x1, y1, x2, y2 = bbox
    if not h_real or not fx or y1 <= 1 or y2 >= img_h - 2 or y2 - y1 < 4:
        return None
    return fx * h_real / float(y2 - y1)


def fuse_range(lidar, size_est):
    """Prefer the lidar unless it disagrees badly with the size estimate
    (then the ray probably hit something else: a door frame, a table edge)."""
    if lidar is None:
        return size_est
    if size_est is None or 0.6 * size_est <= lidar <= 1.6 * size_est:
        return lidar
    return size_est


def detect_colours(rgb, min_area=150):
    """rgb: HxWx3 uint8. Returns [{label, conf, bbox:[x1,y1,x2,y2], source}]."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out = []
    for colour, ranges in HSV_RANGES.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else (mask | m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill = area / float(w * h)
            if colour in SHAPE_CHECK:
                lo, hi, circ_min = SHAPE_CHECK[colour]
                perim = cv2.arcLength(c, True)
                circ = 4 * math.pi * area / (perim * perim) if perim else 0
                if not (lo <= h / float(w) <= hi) or circ < circ_min:
                    continue
            out.append({"label": f"{colour} {COLOUR_SHAPE[colour]}",
                        "conf": round(min(0.99, 0.5 + 0.5 * fill), 2),
                        "bbox": [x, y, x + w, y + h], "source": "colour"})
    return out


class YoloDetector:
    """Ultralytics YOLO (COCO classes). Model weights auto-download once."""

    def __init__(self, model="yolo11n.pt", conf=0.35, device=None):
        import os
        from ultralytics import YOLO
        if os.sep not in model and not os.path.exists(model):
            # keep the weights in ~/.llm_home_robot instead of the current directory
            folder = os.path.expanduser("~/.llm_home_robot")
            os.makedirs(folder, exist_ok=True)
            model = os.path.join(folder, model)
        self.model = YOLO(model)
        self.conf = conf
        self.device = device
        self.names = self.model.names

    def __call__(self, rgb):
        res = self.model.predict(rgb[:, :, ::-1], conf=self.conf, verbose=False,
                                 device=self.device, imgsz=640)[0]   # expects BGR
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = self.names[int(box.cls[0])]
            out.append({"label": "ball" if label == "sports ball" else label,
                        "conf": round(float(box.conf[0]), 2),
                        "bbox": [x1, y1, x2, y2], "source": "yolo"})
        return out


# ---------------------------------------------------------------------------
# Object memory (semantic map)
# ---------------------------------------------------------------------------
class ObjectMemory:
    """Remembers where things were last seen: label -> list of instances.
    Sightings of the same label within merge_dist are the same object."""

    def __init__(self, merge_dist=1.0, max_per_label=5):
        self.items = {}
        self.merge_dist, self.max_per_label = merge_dist, max_per_label

    def add(self, label, x, y, conf, room=None, t=None):
        t = time.time() if t is None else t
        if " " not in label:
            # YOLO's generic "ball" next to a colour detector's "red ball" is the same thing
            for lab, others in self.items.items():
                if lab != label and lab.endswith(" " + label):
                    for o in others:
                        if math.dist((o["x"], o["y"]), (x, y)) < self.merge_dist:
                            label = lab
                            break
        inst = self.items.setdefault(label, [])
        for o in inst:
            if math.dist((o["x"], o["y"]), (x, y)) < self.merge_dist:
                k = 0.3                                  # smooth the position
                o["x"], o["y"] = o["x"] + k * (x - o["x"]), o["y"] + k * (y - o["y"])
                o["conf"], o["seen"], o["room"] = max(o["conf"], conf), t, room or o["room"]
                o["count"] += 1
                return o
        o = {"label": label, "x": x, "y": y, "conf": conf, "room": room, "seen": t, "count": 1}
        inst.append(o)
        inst.sort(key=lambda d: -d["seen"])
        del inst[self.max_per_label:]
        return o

    def find(self, query, max_age=None, now=None, min_count=1):
        """Best remembered instance matching query (most recent first)."""
        now = time.time() if now is None else now
        hits = [o for lab, inst in self.items.items() if matches(query, lab) for o in inst
                if o["count"] >= min_count and (max_age is None or now - o["seen"] <= max_age)]
        return max(hits, key=lambda o: o["seen"]) if hits else None

    def to_list(self):
        return [o for inst in self.items.values() for o in inst]

    def load(self, data):
        for o in data or []:
            self.items.setdefault(o["label"], []).append(dict(o))

    def summary(self, nice=lambda s: s.replace("_", " "), min_count=3):
        """'I've seen: 2 people (bedroom, office); 1 red ball (living room).'
        Things seen fewer than min_count times are left out (likely false hits)."""
        parts = []
        for lab, inst in sorted(self.items.items()):
            inst = [o for o in inst if o["count"] >= min_count]
            if not inst:
                continue
            rooms = sorted({nice(o["room"]) for o in inst if o["room"]})
            parts.append(f"{len(inst)} {plural(lab, len(inst))}"
                         + (f" ({', '.join(rooms)})" if rooms else ""))
        if not parts:
            return "I haven't seen anything I recognise yet."
        return "I've seen: " + "; ".join(parts) + "."


def plural(label, n):
    if n == 1:
        return label
    if label == "person":
        return "people"
    if label.endswith(("x", "s", "ch", "sh")):
        return label + "es"
    return label + "s"

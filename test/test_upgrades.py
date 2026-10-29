"""Tests for the upgrades: perception helpers, follow-me tracker, voice
helpers and the new commands. No ROS, no Gazebo, no microphone needed."""
import math
import os

import numpy as np
import pytest

from llm_home_robot import follow as fl
from llm_home_robot import house_layout as house
from llm_home_robot import perception as pc
from llm_home_robot.command_parser import parse, validate
from llm_home_robot.place_graph import PlaceGraph
from llm_home_robot.voice_utils import Segmenter, clean_transcript

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def g():
    return PlaceGraph.from_files(os.path.join(PKG, "config", "places.yaml"))


# ------------------------------------------------------------------ objects
@pytest.mark.parametrize("text,obj", [
    ("the red ball", "red ball"), ("my cup", "cup"), ("a mug", "cup"), ("someone", "person"),
    ("the sofa", "couch"), ("fridge", "refrigerator"), ("blue thing", "blue box"),
    ("my phone", "cell phone"), ("the TV", "tv"),
])
def test_canonical_object(text, obj):
    assert pc.canonical_object(text) == obj


@pytest.mark.parametrize("query,label,ok", [
    ("ball", "red ball", True), ("ball", "ball", True), ("red ball", "ball", False),
    ("red ball", "red ball", True), ("blue box", "red ball", False), ("person", "person", True),
    ("cup", "bottle", False), ("sofa", "couch", True),
])
def test_matches(query, label, ok):
    assert pc.matches(query, label) is ok


def test_search_order_uses_hints():
    rooms = ["hallway", "office", "bathroom", "kitchen", "bedroom", "living_room", "dining_room"]
    assert pc.search_rooms("cup", rooms)[:2] == ["kitchen", "dining_room"]
    assert pc.search_rooms("toilet", rooms)[0] == "bathroom"
    assert sorted(pc.search_rooms("blue box", rooms)) == sorted(rooms)


def test_bearing_and_world_projection():
    assert pc.bearing_of(320, 640, 1.2) == pytest.approx(0.0)
    assert pc.bearing_of(0, 640, 1.2) == pytest.approx(0.6)          # left edge = +hfov/2
    assert pc.bearing_of(640, 640, 1.2) == pytest.approx(-0.6)
    x, y = pc.to_world((1.0, 2.0, math.pi / 2), 0.0, 3.0, sensor_offset=0.0)
    assert (x, y) == pytest.approx((1.0, 5.0))


def test_range_at_bearing():
    n = 360
    ranges = [5.0] * n
    amin, inc = -math.pi, 2 * math.pi / n
    i = int(round((0.5 - amin) / inc))
    for k in range(i - 2, i + 3):
        ranges[k] = 1.5
    assert pc.range_at_bearing(ranges, amin, inc, 0.5) == pytest.approx(1.5)
    assert pc.range_at_bearing([float("inf")] * n, amin, inc, 0.5) is None


def test_fuse_range_rejects_wrong_lidar_hit():
    est = pc.estimate_range("red ball", [300, 200, 340, 240], 480, 467.7)   # 40 px tall
    assert est == pytest.approx(467.7 * 0.30 / 40)
    assert pc.fuse_range(3.4, est) == 3.4          # agrees -> lidar
    assert pc.fuse_range(0.8, est) == est          # lidar hit a door frame -> size estimate
    assert pc.fuse_range(None, est) == est
    assert pc.estimate_range("red ball", [300, 0, 340, 240], 480, 467.7) is None   # cut off


def _canvas():
    return np.full((240, 320, 3), (200, 190, 175), np.uint8)       # beige floor


def test_colour_detector_finds_ball_ignores_furniture():
    cv2 = pytest.importorskip("cv2")
    img = _canvas()
    cv2.circle(img, (100, 120), 25, (230, 20, 20), -1)              # red ball (RGB)
    cv2.rectangle(img, (200, 60), (260, 200), (90, 75, 140), -1)    # purple-ish sofa (muted)
    cv2.rectangle(img, (270, 100), (300, 150), (5, 40, 60), -1)     # dark teal chair seat
    dets = pc.detect_colours(img)
    assert [d["label"] for d in dets] == ["red ball"]
    x1, y1, x2, y2 = dets[0]["bbox"]
    assert 70 < x1 < 80 and 120 < x2 < 130


def test_colour_detector_rejects_tall_red_can():
    cv2 = pytest.importorskip("cv2")
    img = _canvas()
    cv2.rectangle(img, (100, 60), (125, 130), (220, 15, 20), -1)    # tall red can
    assert pc.detect_colours(img) == []


def test_colour_detector_blue_box():
    cv2 = pytest.importorskip("cv2")
    img = _canvas()
    cv2.rectangle(img, (50, 50), (110, 110), (12, 60, 230), -1)
    assert [d["label"] for d in pc.detect_colours(img)] == ["blue box"]


def test_generic_label_merges_into_coloured_one():
    m = pc.ObjectMemory(merge_dist=1.5)
    m.add("red ball", -8.2, 2.8, 0.9, "living_room", t=1)
    m.add("ball", -8.3, 2.9, 0.8, "living_room", t=2)        # YOLO's view of the same ball
    m.add("ball", 5.0, 5.0, 0.8, "dining_room", t=3)         # a different ball
    assert set(m.items) == {"red ball", "ball"}
    assert m.items["red ball"][0]["count"] == 2 and len(m.items["ball"]) == 1


def test_rememberable():
    assert pc.rememberable("person") and pc.rememberable("red ball") and pc.rememberable("cup")
    assert not pc.rememberable("airplane") and not pc.rememberable("train")


def test_object_memory():
    m = pc.ObjectMemory()
    m.add("cup", 1.0, 5.0, 0.6, "kitchen", t=100)
    m.add("cup", 1.2, 5.1, 0.8, "kitchen", t=110)      # same cup
    m.add("cup", 7.8, 5.3, 0.7, "dining_room", t=105)  # a second cup
    assert len(m.items["cup"]) == 2
    best = m.find("mug", now=120)
    assert best["room"] == "kitchen" and best["count"] == 2 and best["conf"] == 0.8
    assert m.find("cup", min_count=2, now=120)["room"] == "kitchen"
    assert m.find("person") is None
    assert "1 cup (kitchen)" in m.summary(min_count=2)   # the dining-room cup was seen once only
    assert "2 cups" in m.summary(min_count=1)
    m.add("blue box", 0, 0, 0.9, "office", t=1)
    m.add("blue box", 5, 5, 0.9, "office", t=2)
    for _ in range(2):
        m.add("person", 3, 3, 0.9, "bedroom", t=3)
        m.add("person", 9, -3, 0.9, "office", t=4)
    assert "2 people (bedroom, office)" in m.summary(min_count=2)
    assert "2 blue boxes" in m.summary(min_count=1)
    m2 = pc.ObjectMemory()
    m2.load(m.to_list())
    assert len(m2.to_list()) == len(m.to_list())


# ------------------------------------------------------------------ follow
def _scan_with(objs, n=360, wall=6.0):
    """Synthetic scan: circles (x, y, r) in the base frame in front of a far wall."""
    amin, inc = -math.pi, 2 * math.pi / n
    ranges = []
    for i in range(n):
        a = amin + i * inc
        best = wall
        dx, dy = math.cos(a), math.sin(a)
        for cx, cy, r in objs:
            cx -= 0.05                                   # lidar is 5 cm ahead of base
            b = dx * cx + dy * cy
            c = cx * cx + cy * cy - r * r
            disc = b * b - c
            if disc >= 0 and b - math.sqrt(disc) > 0:
                best = min(best, b - math.sqrt(disc))
        ranges.append(best)
    return ranges, amin, inc


def test_two_legs_make_one_person():
    ranges, amin, inc = _scan_with([(1.5, 0.1, 0.07), (1.5, -0.1, 0.07)])
    cands = fl.candidates(fl.scan_points(ranges, amin, inc, rmax=5.5))
    assert len(cands) == 1
    assert cands[0][0] == pytest.approx(1.45, abs=0.08) and abs(cands[0][1]) < 0.05


def test_walls_are_not_people():
    ranges, amin, inc = _scan_with([])
    assert fl.candidates(fl.scan_points(ranges, amin, inc, rmax=10)) == []


def test_tracker_follows_moving_person_and_ignores_decoy():
    tr = fl.PersonTracker()
    pose = (0.0, 0.0, 0.0)
    person = [1.5, 0.0]
    decoy = (1.0, 1.2, 0.15)
    ranges, amin, inc = _scan_with([(person[0], person[1], 0.2), decoy])
    assert tr.acquire(fl.candidates(fl.scan_points(ranges, amin, inc, rmax=5.5)), pose,
                      prefer_bearing=0.0)
    t = 0.0
    for _ in range(30):                     # person walks left-forward, decoy stays
        person[0] += 0.03
        person[1] += 0.04
        t += 0.1
        ranges, amin, inc = _scan_with([(person[0], person[1], 0.2), decoy])
        got = tr.update(fl.candidates(fl.scan_points(ranges, amin, inc, rmax=5.5)), pose, t)
        assert got is not None
    assert got[0] == pytest.approx(person[0] - 0.2, abs=0.15)
    assert got[1] == pytest.approx(person[1], abs=0.15)
    assert not tr.lost(t + 1.0) and tr.lost(t + 3.5)


def test_follow_command():
    v, w = fl.follow_command((2.0, 0.0), distance=0.9)
    assert v > 0.3 and w == 0
    v, w = fl.follow_command((0.9, 0.0), distance=0.9)
    assert v == 0
    v, w = fl.follow_command((0.0, 1.5), distance=0.9)     # beside us: turn first
    assert v == 0 and w > 0
    v, _ = fl.follow_command((0.5, 0.0), distance=0.9)     # too close: back off a little
    assert v < 0


# ------------------------------------------------------------------- voice
def _frames(seconds, amp, rng):
    n = int(16000 * seconds)
    t = np.arange(n) / 16000
    sig = amp * np.sin(2 * np.pi * 220 * t) if amp else np.zeros(n)
    return (sig + rng.normal(0, 0.002, n)).astype(np.float32)


def test_segmenter_cuts_utterances():
    rng = np.random.default_rng(1)
    stream = np.concatenate([_frames(1.0, 0, rng), _frames(1.2, 0.2, rng), _frames(1.0, 0, rng),
                             _frames(0.05, 0.2, rng),                       # click: too short
                             _frames(1.0, 0, rng), _frames(0.8, 0.2, rng), _frames(1.0, 0, rng)])
    seg, out = Segmenter(), []
    for i in range(0, len(stream) - 480, 480):
        u = seg.feed(stream[i:i + 480])
        if u is not None:
            out.append(len(u) / 16000)
    assert len(out) == 2
    assert 1.2 < out[0] < 2.6 and 0.8 < out[1] < 2.2


@pytest.mark.parametrize("raw,wake,expect", [
    (" Go to the kitchen.", "", "Go to the kitchen"),
    ("Thank you.", "", None), ("[Music]", "", None), ("you", "", None),
    ("Robot, go to the kitchen.", "robot", "go to the kitchen"),
    ("Hey robot follow me", "robot", "follow me"),
    ("Go to the kitchen.", "robot", None),
    ("Robot.", "robot", None),
])
def test_clean_transcript(raw, wake, expect):
    assert clean_transcript(raw, wake) == expect


# ------------------------------------------------------------ new commands
@pytest.mark.parametrize("text,expected", [
    ("follow me", [{"type": "follow"}]),
    ("come with me", [{"type": "follow"}]),
    ("stop following me", [{"type": "stop"}]),
    ("find the red ball", [{"type": "find", "object": "red ball"}]),
    ("look for my mug", [{"type": "find", "object": "cup"}]),
    ("find someone", [{"type": "find", "object": "person"}]),
    ("is anyone in the kitchen", [{"type": "find", "object": "person", "room": "kitchen"}]),
    ("where is the cup", [{"type": "locate", "object": "cup"}]),
    ("have you seen a toilet", [{"type": "locate", "object": "toilet"}]),
    ("have you seen my phone", [{"type": "locate", "object": "cell phone"}]),
    ("what do you see", [{"type": "look"}]),
    ("patrol the house", [{"type": "patrol"}]),
    ("explore", [{"type": "patrol"}]),
    ("forget what you've seen", [{"type": "forget"}]),
    ("clear your memory", [{"type": "forget"}]),
    ("go to the red ball", [{"type": "find", "object": "red ball"}]),
    ("find the kitchen", [{"type": "goto", "place": "kitchen"}]),
    ("go to the sofa", [{"type": "goto", "place": "sofa"}]),
])
def test_parse_new_commands(g, text, expected):
    assert parse(text, g) == expected


def test_parse_mixed_sequence(g):
    assert parse("go to the bedroom then find the blue box", g) == [
        {"type": "goto", "place": "bedroom"}, {"type": "find", "object": "blue box"}]


def test_unknown_things_go_to_llm(g):
    assert parse("find the garden shed", g) is None
    assert parse("where is the kitchen", g) is None


def test_validate_llm_find(g):
    out = validate([{"type": "find", "object": "my Mug", "room": "Kitchen"},
                    {"type": "follow"}, {"type": "patrol"}], g)
    assert out == [{"type": "find", "object": "cup", "room": "kitchen"},
                   {"type": "follow"}, {"type": "patrol"}]
    with pytest.raises(ValueError):
        validate([{"type": "find"}], g)
    with pytest.raises(ValueError):
        validate([{"type": "scan", "object": "cup"}], g)     # internal action, not for the LLM


# ------------------------------------------------------------- world + map
def test_objects_and_props_clear_of_routes(g):
    # generate_world.validate covers this too; keep a fast check here
    for name, *box in house.obstacles():
        for a, nbrs in g.adj.items():
            for b in nbrs:
                d = house.segment_box_dist(*g.nodes[a][:2], *g.nodes[b][:2], box, step=0.05)
                assert d >= house.ROBOT_R - 1e-6, (name, a, b, d)


def test_generated_map_matches_layout():
    pgm = os.path.join(PKG, "maps", "big_home.pgm")
    with open(pgm, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        assert int(f.readline()) == 255
        data = np.frombuffer(f.read(), np.uint8).reshape(h, w)

    def cell(x, y):
        return data[h - 1 - int((y + 7.5) / 0.05), int((x + 10.5) / 0.05)]
    assert cell(0.0, 0.0) == 254          # hallway is free
    assert cell(-6.0, 6.3) == 0           # sofa is occupied
    assert cell(-6.0, 1.0) == 254         # living-room doorway is open
    assert cell(-4.0, 1.0) == 0           # wall next to it
    assert cell(-8.3, 2.8) == 0           # red ball
    assert cell(-10.4, 7.4) == 205        # outside the house: unknown


# ------------------------------------------------------------------- simple LLM
def test_extract_json_handles_chatty_models():
    from llm_home_robot.llm_client import extract_json
    assert extract_json('```json\n{"reply": "hi", "actions": []}\n```') == {"reply": "hi", "actions": []}
    assert extract_json('Sure! {"actions": [{"type": "stop"}]} hope that helps')["actions"] == [{"type": "stop"}]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_prompt_lists_places_and_camera_actions(g):
    from llm_home_robot.llm_client import CAMERA, PROMPT
    s = PROMPT.format(places=", ".join(g.place_names()), room="hallway", camera=CAMERA.format())
    assert "fridge" in s and "in the hallway" in s and '"type": "find"' in s and "{{" not in s
    s2 = PROMPT.format(places="kitchen", room="hallway", camera="")
    assert '"type": "find"' not in s2


def test_validate_rejects_bad_llm_output(g):
    for bad in ([{"type": "goto", "place": "garden"}], [{"type": "fly"}],
                [{"type": "wait", "seconds": 1}] * 13, "go to kitchen"):
        with pytest.raises(ValueError):
            validate(bad, g)

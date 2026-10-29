"""Tests for the RViz marker builder (no ROS needed)."""
import os

import pytest

from llm_home_robot import house_layout as house
from llm_home_robot import viz_markers as vm
from llm_home_robot.place_graph import PlaceGraph

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = {"cube", "sphere", "cylinder", "arrow", "text", "line_strip", "line_list", "sphere_list"}


@pytest.fixture()
def g():
    return PlaceGraph.from_files(os.path.join(PKG, "config", "places.yaml"))


def check(markers):
    keys = [(m["ns"], m["id"]) for m in markers]
    assert len(keys) == len(set(keys)), "duplicate ns/id would overwrite markers in RViz"
    for m in markers:
        assert m["type"] in KINDS
        assert len(m["rgba"]) == 4 and all(0 <= c <= 1 for c in m["rgba"])
        if m["type"] == "text":
            assert m["text"] and m["text"].isascii()
        if m["type"] in ("line_strip", "line_list"):
            assert len(m["points"]) >= 2
        if m["type"] == "line_list":
            assert len(m["points"]) % 2 == 0


def test_house(g):
    ms = vm.house_markers(g.rooms)
    check(ms)
    ns = [m["ns"] for m in ms]
    assert ns.count("walls") == len(house.wall_boxes())
    assert ns.count("furniture") == len(house.FURNITURE)
    assert ns.count("floors") == ns.count("room_names") == len(g.rooms)
    names = {m["text"] for m in ms if m["ns"] == "room_names"}
    assert "KITCHEN" in names and "LIVING ROOM" in names


def test_room_names_inside_their_room(g):
    pins = [(q["x"], q["y"]) for q in g.places.values()]
    for m in vm.house_markers(g.rooms, places=pins):
        if m["ns"] == "room_names":
            room = m["text"].lower().replace(" ", "_")
            assert g.room_at(m["x"], m["y"]) == room


def test_places_and_goal_highlight(g):
    ms = vm.place_markers(g, goal="kitchen")
    check(ms)
    labels = [m for m in ms if m["ns"] == "place_names"]
    assert len(labels) == len(g.places)
    kitchen = next(m for m in labels if m["text"].startswith("kitchen"))
    assert kitchen["rgba"] == vm.GOAL_TEXT_RGBA
    living = next(m for m in labels if m["text"].startswith("living room"))
    assert "living room)" not in living["text"]   # alias equal to the name is dropped
    sofa = next(m for m in labels if m["text"].startswith("sofa"))
    assert "couch" in sofa["text"]         # alias shown


def test_learned_place_is_purple(g):
    g.add_place("reading corner", -8.0, 2.0)
    pins = vm.place_markers(g, learned={"reading_corner"})
    idx = sorted(g.places).index("reading_corner")
    pin = next(m for m in pins if m["ns"] == "place_pins" and m["id"] == idx)
    assert pin["rgba"] == vm.LEARNED_RGBA


def test_graph(g):
    ms = vm.graph_markers(g)
    check(ms)
    lines = ms[0]["points"]
    n_edges = sum(len(v) for v in g.adj.values()) // 2
    assert len(lines) == 2 * n_edges


def test_robot_busy_with_route():
    status = {"state": "busy", "goal": "kitchen", "room": "hallway", "x": 0.0, "y": 0.0,
              "route": [[1.0, 0.0], [1.0, 2.0], [1.0, 3.5]]}
    ms = vm.robot_markers(status, "Going to the kitchen via hallway -> kitchen (3.5 m).")
    check(ms)
    by_ns = {m["ns"]: m for m in ms}
    assert "-> kitchen" in by_ns["robot_status"]["text"]
    assert len(by_ns["planned_route"]["points"]) == 4
    assert by_ns["planned_route"]["rgba"][3] == 1.0
    assert by_ns["speech"]["rgba"][3] == 1.0


def test_robot_idle_hides_route_and_speech():
    ms = vm.robot_markers({"state": "idle", "room": "kitchen", "x": 1, "y": 3.5, "route": []})
    check(ms)
    by_ns = {m["ns"]: m for m in ms}
    assert by_ns["planned_route"]["rgba"][3] == 0
    assert by_ns["speech"]["rgba"][3] == 0
    assert vm.robot_markers(None) == []


def test_speech_degrees_and_wrap():
    ms = vm.robot_markers({"state": "idle", "x": 0, "y": 0},
                          "OK - turn left 90°, then go to bedroom, then turn around and come back home.")
    speech = next(m for m in ms if m["ns"] == "speech")
    assert "90 deg" in speech["text"] and speech["text"].isascii()
    assert all(len(line) <= 40 for line in speech["text"].split("\n"))


def test_click_nearest_place(g):
    assert vm.nearest_place(g, 1.2, 3.3) == "kitchen"
    assert vm.nearest_place(g, 3.3, 5.0) == "fridge"
    assert vm.nearest_place(g, 0.0, 6.9, max_dist=0.5) is None


def test_room_names_avoid_furniture(g):
    pins = [(q["x"], q["y"]) for q in g.places.values()]
    for m in vm.house_markers(g.rooms, places=pins):
        if m["ns"] != "room_names":
            continue
        for _n, fx, fy, fsx, fsy, _h, _c in house.FURNITURE:
            assert not (abs(m["x"] - fx) < fsx / 2 and abs(m["y"] - fy) < fsy / 2), (m["text"], _n)

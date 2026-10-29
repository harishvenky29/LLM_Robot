"""Unit tests for the place graph and the rule-based command parser.
Run with:  python3 -m pytest test/ -v   (no ROS needed)"""
import math
import os

import pytest

from llm_home_robot.command_parser import parse, validate, describe
from llm_home_robot.place_graph import PlaceGraph

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def g():
    return PlaceGraph.from_files(os.path.join(PKG, "config", "places.yaml"))


# ------------------------------------------------------------- place graph
def test_rooms(g):
    assert g.room_at(0, 0) == "hallway"
    assert g.room_at(1, 3.5) == "kitchen"
    assert g.room_at(-6, -3.5) == "bedroom"
    assert g.room_at(5.5, -3.5) == "office"


@pytest.mark.parametrize("text,place", [
    ("kitchen", "kitchen"), ("the lounge", "living_room"), ("Living Room", "living_room"),
    ("kitchn", "kitchen"), ("loo", "bathroom"), ("the fridge", "fridge"),
    ("charging dock", "charging_station"), ("bathroom door", "bathroom"),
])
def test_resolve(g, text, place):
    assert g.resolve(text) == place


def test_resolve_unknown(g):
    assert g.resolve("garden shed") is None


def test_every_place_reachable_from_everywhere(g):
    for a in g.places:
        for b in g.places:
            assert g.shortest_path(a, b) is not None, (a, b)


def test_route_goes_through_doors(g):
    names, pts, yaw = g.plan(0.0, 0.0, "kitchen")
    assert names == ["hall_p1", "kitchen_door", "kitchen"]
    assert yaw == pytest.approx(math.pi / 2)


def test_bedroom_to_dining_uses_hallway(g):
    names, _, _ = g.plan(-6.0, -3.5, "dining_room")
    assert "bedroom_door" in names and "dining_door" in names


def test_living_to_kitchen_uses_side_door(g):
    names, _, _ = g.plan(-6.0, 3.5, "kitchen")
    assert "lk_living" in names  # shorter than going via hallway


def test_learned_place(g, tmp_path):
    name = g.add_place("reading corner", -8.0, 2.0)
    assert name == "reading_corner"
    assert g.resolve("reading corner") == "reading_corner"
    assert g.plan(0, 0, "reading_corner") is not None


# ----------------------------------------------------------------- parser
@pytest.mark.parametrize("text,expected", [
    ("move forward", [{"type": "move", "distance": 1.0}]),
    ("go forward 2 meters", [{"type": "move", "distance": 2.0}]),
    ("move forward 1.5 m", [{"type": "move", "distance": 1.5}]),
    ("drive ahead 50 cm", [{"type": "move", "distance": 0.5}]),
    ("go forward a bit", [{"type": "move", "distance": 0.3}]),
    ("move forward three metres", [{"type": "move", "distance": 3.0}]),
    ("back up 1 meter", [{"type": "move", "distance": -1.0}]),
    ("reverse", [{"type": "move", "distance": -1.0}]),
    ("turn left", [{"type": "turn", "angle": 90.0}]),
    ("turn right 45 degrees", [{"type": "turn", "angle": -45.0}]),
    ("rotate left 30°", [{"type": "turn", "angle": 30.0}]),
    ("turn around", [{"type": "turn", "angle": 180.0}]),
    ("stop", [{"type": "stop"}]),
    ("where are you?", [{"type": "where"}]),
    ("what places do you know", [{"type": "list_places"}]),
    ("go to the kitchen", [{"type": "goto", "place": "kitchen"}]),
    ("Please take me to the bathroom", [{"type": "goto", "place": "bathroom"}]),
    ("head to the lounge", [{"type": "goto", "place": "living_room"}]),
    ("kitchen", [{"type": "goto", "place": "kitchen"}]),
    ("go back to the kitchen", [{"type": "goto", "place": "kitchen"}]),
    ("go home", [{"type": "goto", "place": "charging_station"}]),
    ("wait 3 seconds", [{"type": "wait", "seconds": 3.0}]),
    ("remember this place as reading corner",
     [{"type": "save_place", "name": "reading corner"}]),
])
def test_parse(g, text, expected):
    assert parse(text, g) == expected


def test_parse_sequence(g):
    acts = parse("go forward 2 meters then turn left and go to the bedroom", g)
    assert acts == [
        {"type": "move", "distance": 2.0},
        {"type": "turn", "angle": 90.0},
        {"type": "goto", "place": "bedroom"},
    ]
    assert describe(acts) == "move forward 2.00 m, then turn left 90°, then go to bedroom"


def test_save_place_name_with_and(g):
    assert parse("save this spot as bread and butter corner", g) == [
        {"type": "save_place", "name": "bread and butter corner"}]


def test_parse_unknown_goes_to_llm(g):
    assert parse("I'm hungry", g) is None
    assert parse("go to the garden shed", g) is None


def test_validate_clamps_and_resolves(g):
    out = validate([{"type": "move", "distance": 100},
                    {"type": "goto", "place": "Living Room"}], g)
    assert out == [{"type": "move", "distance": 8.0},
                   {"type": "goto", "place": "living_room"}]
    with pytest.raises(ValueError):
        validate([{"type": "goto", "place": "moon"}], g)
    with pytest.raises(ValueError):
        validate([{"type": "fly"}], g)

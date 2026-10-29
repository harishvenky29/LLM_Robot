#!/usr/bin/env python3
"""Check how well the LLM turns requests into robot actions (no ROS needed,
just Ollama - or Claude with --backend anthropic).

    python3 tools/eval_llm.py
    python3 tools/eval_llm.py --model llama3.2:3b

Each request goes through exactly what the robot does: LLM -> validate().
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llm_home_robot.command_parser import describe, validate  # noqa: E402
from llm_home_robot.llm_client import LLMClient  # noqa: E402
from llm_home_robot.place_graph import PlaceGraph  # noqa: E402


def place_list(g):
    out = []
    for n in g.place_names():
        al = [a for a in g.places[n]["aliases"] if a != n.replace("_", " ")][:3]
        out.append(f"{n} ({', '.join(al)})" if al else n)
    return out


def gotos(p):
    return [a["place"] for a in p if a["type"] == "goto"]


# request, check(plan, reply)
CASES = [
    ("go to the kitchen", lambda p, r: gotos(p) == ["kitchen"]),
    ("take me to the loo", lambda p, r: gotos(p) == ["bathroom"]),
    ("move forward two metres then turn right", lambda p, r:
        [a["type"] for a in p] == ["move", "turn"] and p[0]["distance"] > 0 and p[1]["angle"] < 0),
    ("go to the bedroom, wait 5 seconds and come back to the hallway", lambda p, r:
        gotos(p) == ["bedroom", "hallway"] and any(a["type"] == "wait" for a in p)),
    ("I'm really thirsty", lambda p, r: gotos(p)[:1] in (["fridge"], ["kitchen"])),
    ("I'm so tired, I need to lie down", lambda p, r: gotos(p)[:1] in (["bed"], ["bedroom"])),
    ("I need to answer some emails", lambda p, r: gotos(p)[:1] in (["desk"], ["office"])),
    ("someone is ringing the doorbell", lambda p, r: gotos(p) == ["front_door"]),
    ("your battery is getting low", lambda p, r: gotos(p) == ["charging_station"]),
    ("I want to watch some TV", lambda p, r: gotos(p)[:1] in (["sofa"], ["living_room"])),
    ("can you find the blue box for me", lambda p, r: [a["type"] for a in p] == ["find"]),
    ("keep me company while I walk around", lambda p, r: [a["type"] for a in p] == ["follow"]),
    ("what's the weather like in York?", lambda p, r: p == [] and r),
    ("go home", lambda p, r: gotos(p) == ["charging_station"]),
    ("how far are you from the bedroom?", lambda p, r: not gotos(p)),
    ("go for a petrol", lambda p, r: [a["type"] for a in p] == ["patrol"]),
    ("is there anyone in the dining room", lambda p, r: [(a["type"], a.get("room")) for a in p] == [("find", "dining_room")]),
    ("how many people are in the house?", lambda p, r: [a["type"] for a in p] == ["patrol"]),
    ("go to the garden", lambda p, r: p == [] and r),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    llm = LLMClient(a.backend, a.model, timeout=180)
    g = PlaceGraph.from_files(os.path.join(ROOT, "config", "places.yaml"))
    print(f"LLM eval: {a.backend} / {llm.model}\n")
    passed, times = 0, []
    for text, check in CASES:
        t0 = time.time()
        try:
            raw, reply = llm.interpret(text, place_list(g), "hallway", camera=True)
            plan = validate(raw, g) if raw else []
            ok = bool(check(plan, reply))
            note = (describe(plan) if plan else "-") + f'  | "{reply}"'
        except Exception as e:  # noqa: BLE001
            ok, note = False, f"ERROR {e.__class__.__name__}: {e}"
        times.append(time.time() - t0)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {times[-1]:5.1f}s  {text}\n      -> {note[:200]}")
    print(f"\nScore: {passed}/{len(CASES)}   avg {sum(times) / len(times):.1f}s per request")


if __name__ == "__main__":
    main()

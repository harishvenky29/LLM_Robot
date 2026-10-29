"""Rule-based parser for basic robot commands.

Turns text like "go forward 2 metres then turn left and go to the kitchen"
into a list of action dicts the executor understands:

    {"type": "move", "distance": 2.0}          metres, negative = backwards
    {"type": "turn", "angle": 90.0}            degrees, positive = left (CCW)
    {"type": "goto", "place": "kitchen"}
    {"type": "wait", "seconds": 3}
    {"type": "stop"} | {"type": "where"} | {"type": "list_places"}
    {"type": "save_place", "name": "reading corner"}
    {"type": "follow"}                         follow the nearest person (lidar)
    {"type": "find", "object": "red ball"}     search rooms with the camera, drive to it
    {"type": "look"}                           describe what the camera sees
    {"type": "locate", "object": "cup"}        where was it last seen?
    {"type": "patrol"}                         visit every room, catalogue objects
    {"type": "forget"}                         clear the object memory

parse() returns None if any part is not understood - the interpreter then
hands the whole sentence to the LLM. Pure Python, no ROS.
"""
import re

from .perception import SYNONYMS, COLOURS, canonical_object

ACTION_TYPES = {"move", "turn", "goto", "wait", "stop", "where", "list_places", "save_place",
                "follow", "find", "look", "locate", "patrol", "forget"}
KNOWN_OBJECTS = set(SYNONYMS.values())


def known_object(text):
    """Canonical object name if text names something the camera could find."""
    obj = canonical_object(text)
    if obj and (obj in KNOWN_OBJECTS or obj.split()[0] in COLOURS):
        return obj
    return None

NUM_WORDS = {
    "zero": 0, "half": 0.5, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty five": 45, "forty": 40,
    "sixty": 60, "ninety": 90, "hundred eighty": 180, "one eighty": 180,
}
NUM = r"(\d+(?:\.\d+)?|\b(?:" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")\b)"

SPLIT_RE = re.compile(r"\s*(?:,\s*)?\b(?:and then|then|after that|and|afterwards)\b\s*|\s*[,;]\s*|\.(?:\s+|$)")
FILLER_RE = re.compile(r"\b(please|robot|hey|ok|okay|now|can you|could you|would you|kindly|for me)\b")


def _num(tok):
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(NUM_WORDS.get(tok, 1))


def _clean(text):
    text = text.lower().replace("°", " degrees ")
    text = FILLER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ?!")


def _distance(seg, default=1.0):
    m = re.search(NUM + r"\s*(cm|centimet(?:er|re)s?|m\b|met(?:er|re)s?|steps?|feet|foot|ft)?", seg)
    small = re.search(r"\b(a (little )?bit|slightly|a little|a touch)\b", seg)
    if small and not re.search(r"\d", seg):
        return 0.3
    if not m:
        return default
    value, unit = _num(m.group(1)), (m.group(2) or "m")
    if unit.startswith("c"):
        value /= 100.0
    elif unit.startswith("step"):
        value *= 0.5
    elif unit.startswith(("f", "ft")):
        value *= 0.3048
    return value


def _angle(seg, default=90.0):
    if re.search(r"\b(a (little )?bit|slightly|a little)\b", seg) and not re.search(r"\d", seg):
        return 30.0
    m = re.search(NUM + r"\s*(deg(?:ree)?s?|°)", seg)
    if m:
        return _num(m.group(1))
    if re.search(r"\bhalf (a )?(turn|circle)\b", seg):
        return 180.0
    if re.search(r"\b(full|complete) (turn|circle|spin|rotation)\b|\b360\b", seg):
        return 360.0
    return default


def parse_segment(seg, graph):
    seg = seg.strip()
    if not seg or re.fullmatch(r"(go|come|head|run|hurry|quickly)( on| please)?", seg):
        return []                    # "go and look for someone": the "go" is not a move

    if re.fullmatch(r"(stop|halt|freeze|stay|pause|wait there|stay there|stand still|emergency stop|abort|cancel)"
                    r"( (moving|now|it|that|everything|following( me)?|searching|looking|patrolling|here))?", seg):
        return [{"type": "stop"}]

    if re.search(r"\b(forget|clear|reset|wipe)\b.*\b(seen|objects|things|memory)\b", seg):
        return [{"type": "forget"}]

    if re.search(r"\b(follow|come with|come along with|walk with|stay with|track) (me|us|him|her|them|the person)\b"
                 r"|^follow$|\bfollow me\b", seg):
        return [{"type": "follow"}]

    if re.search(r"\b(patrol|explore|tour|map|scout|inspect|check|search|look (around|through|in)) "
                 r"(the |every |all |each )?(whole |entire )?(house|home|rooms|room|flat|place|everywhere)\b"
                 r"|^(patrol|explore|go exploring|go on patrol|do a patrol|do a tour)$", seg) \
            and not re.search(r"\b(for|find)\b", seg):
        return [{"type": "patrol"}]
    if re.search(r"\bhow many (people|persons|humans)\b|\bcount (the )?(people|persons)\b", seg):
        return [{"type": "patrol"}]               # the patrol report counts people

    if re.search(r"\bwhat (do|can) you see\b|\bwhat('s| is) (in front of you|ahead|around you|there)\b"
                 r"|\b(look around|describe what you see|what are you looking at|what's around)\b", seg):
        return [{"type": "look"}]

    if re.search(r"\b(where are you|where am i|your location|what room|which room|current location|where you are)\b", seg):
        return [{"type": "where"}]

    if re.search(r"\b(list|what|which)\b.*\b(places|rooms|locations)\b|\bwhere can you go\b|\bwhat do you know\b", seg):
        return [{"type": "list_places"}]

    m = re.search(r"\b(?:remember|save|mark|store)\b (?:this|here|current)?\s*(?:place|spot|location|position)?\s*as (.+)|\bcall (?:this|here) (?:place |spot )?(.+)", seg)
    if m:
        name = (m.group(1) or m.group(2)).strip()
        return [{"type": "save_place", "name": name}]

    m = re.search(r"\b(?:wait|pause|sleep)\b (?:for )?" + NUM + r"\s*(s|sec|secs|seconds?|minutes?|mins?)?", seg)
    if m:
        secs = _num(m.group(1)) * (60 if (m.group(2) or "").startswith("m") else 1)
        return [{"type": "wait", "seconds": secs}]

    if re.search(r"\b(turn around|u ?turn|about face|face the other way|spin around)\b", seg):
        return [{"type": "turn", "angle": 180.0}]

    m = re.search(r"\b(turn|rotate|spin|face|look)\b.*\b(left|right)\b|\b(left|right) turn\b", seg)
    if m:
        side = m.group(2) or m.group(3)
        a = _angle(seg)
        return [{"type": "turn", "angle": a if side == "left" else -a}]

    if re.search(r"\b(spin|rotate)\b", seg) and not re.search(r"\bto\b", seg):
        return [{"type": "turn", "angle": _angle(seg, 360.0)}]

    if re.search(r"\b(back ?up|reverse|backwards?|go back(?! to)|move back|step back|drive back)\b", seg) \
            and not re.search(r"\b(to|into)\b", seg):
        return [{"type": "move", "distance": -abs(_distance(seg))}]

    if re.search(r"\b(forward|forwards|ahead|straight|onward)\b", seg) or \
            re.fullmatch(r"(move|go|drive|advance|step)( " + NUM + r".*)?", seg):
        return [{"type": "move", "distance": abs(_distance(seg))}]

    if re.search(r"\b(go|come|return|head|get)\b ?(back )?home\b|\bgo charge\b|\b(recharge|charge yourself|dock)\b", seg):
        return [{"type": "goto", "place": "charging_station"}]

    m = re.search(r"\b(?:is|are) (?:there )?(.+?) (?:in|inside|at) (.+)", seg)
    if m and graph is not None:
        who = m.group(1)
        who = "person" if re.fullmatch(r"(anyone|anybody|someone|somebody|any ?one|people|anybody home)", who) else who
        obj, place = known_object(who), graph.resolve(m.group(2))
        if obj and place and place in graph.rooms:
            return [{"type": "find", "object": obj, "room": place}]

    m = re.search(r"\b(?:where(?:'s| is| are)|have you seen|did you see|seen) (.+)", seg)
    if m:
        target = m.group(1)
        if not known_object(target) and graph is not None and graph.resolve(target):
            return None                  # "where is the kitchen" - let the LLM explain
        obj = known_object(target) or canonical_object(target)
        return [{"type": "locate", "object": obj}] if obj else None

    m = re.search(r"\b(?:find|look for|search for|search|locate|fetch|seek|hunt for|bring me|get me|show me|"
                  r"where can i find)\b (.+)", seg)
    if m:
        target = m.group(1)
        place = graph.resolve(target) if graph is not None else None
        obj = known_object(target)
        if obj:
            return [{"type": "find", "object": obj}]
        if place:
            return [{"type": "goto", "place": place}]
        return None                      # unknown thing - let the LLM decide

    m = re.search(r"\b(?:go|move|drive|navigate|head|walk|come|take me|bring me|get|run|travel|return|proceed)\b"
                  r"(?: back)?(?: over)?(?: (?:to|into|towards?|in))?(?: the)? (.+)", seg)
    target = m.group(1) if m else seg
    place = graph.resolve(target) if graph is not None else None
    if place:
        return [{"type": "goto", "place": place}]
    obj = known_object(target) if m else None
    if obj:                              # "go to the red ball" - not a place, but findable
        return [{"type": "find", "object": obj}]

    return None  # not understood


def parse(text, graph):
    """Parse a whole utterance. Returns list of actions, or None."""
    text = _clean(text)
    if not text:
        return None
    # protect save_place names from being split on "and"
    m = re.search(r"\b(remember|save|mark|store|call)\b", text)
    if m:
        head, tail = text[:m.start()], text[m.start():]
        parts = [p for p in SPLIT_RE.split(head) if p.strip()] + [tail]
    else:
        parts = [p for p in SPLIT_RE.split(text) if p.strip()]
    actions = []
    for part in parts:
        res = parse_segment(part, graph)
        if res is None:
            return None
        actions.extend(res)
    return actions or None


def describe(actions):
    """Human-readable summary of an action list."""
    out = []
    for a in actions:
        t = a["type"]
        if t == "move":
            d = a["distance"]
            out.append(f"move {'forward' if d >= 0 else 'back'} {abs(d):.2f} m")
        elif t == "turn":
            ang = a["angle"]
            out.append(f"turn {'left' if ang >= 0 else 'right'} {abs(ang):.0f}°")
        elif t == "goto":
            out.append(f"go to {a['place'].replace('_', ' ')}")
        elif t == "wait":
            out.append(f"wait {a['seconds']:.0f} s")
        elif t == "save_place":
            out.append(f"remember this spot as '{a['name']}'")
        elif t == "find":
            where = f" in the {a['room'].replace('_', ' ')}" if a.get("room") else ""
            out.append(f"look for {'anyone' if a['object'] == 'person' else 'the ' + a['object']}{where}")
        elif t == "locate":
            out.append(f"recall where the {a['object']} is")
        elif t == "follow":
            out.append("follow you")
        elif t == "patrol":
            out.append("patrol every room")
        elif t == "look":
            out.append("describe what I see")
        else:
            out.append(t.replace("_", " "))
    return ", then ".join(out)


def validate(actions, graph, max_move=8.0, max_actions=12):
    """Sanitise actions from any source (rules or LLM). Raises ValueError."""
    if not isinstance(actions, list) or not actions:
        raise ValueError("no actions")
    if len(actions) > max_actions:
        raise ValueError(f"too many actions ({len(actions)})")
    clean = []
    for a in actions:
        if not isinstance(a, dict) or a.get("type") not in ACTION_TYPES:
            raise ValueError(f"bad action: {a!r}")
        t = a["type"]
        if t == "move":
            d = float(a.get("distance", 1.0))
            clean.append({"type": t, "distance": max(-max_move, min(max_move, d))})
        elif t == "turn":
            ang = float(a.get("angle", 90.0))
            clean.append({"type": t, "angle": max(-720.0, min(720.0, ang))})
        elif t == "goto":
            place = graph.resolve(re.sub(r"\(.*?\)", " ", str(a.get("place", ""))))
            if not place:
                raise ValueError(f"unknown place: {a.get('place')!r}")
            clean.append({"type": t, "place": place})
        elif t == "wait":
            clean.append({"type": t, "seconds": max(0.0, min(60.0, float(a.get("seconds", 1))))})
        elif t == "save_place":
            name = str(a.get("name", "")).strip()
            if not name:
                raise ValueError("save_place needs a name")
            clean.append({"type": t, "name": name})
        elif t in ("find", "locate"):
            obj = canonical_object(str(a.get("object", "")))
            if not obj:
                raise ValueError(f"{t} needs an object")
            act = {"type": t, "object": obj}
            room = graph.resolve(str(a.get("room", ""))) if t == "find" and a.get("room") else None
            if room and room in graph.rooms:
                act["room"] = room
            clean.append(act)
        else:
            clean.append({"type": t})
    return clean

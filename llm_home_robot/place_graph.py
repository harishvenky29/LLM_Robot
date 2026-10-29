"""The robot's knowledge of the house: rooms, named places and routes.

Pure Python (no ROS imports) so it can be unit-tested anywhere.
"""
import difflib
import heapq
import math
import os
import re

import yaml


def _norm(text):
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(the|a|an|my|our)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class PlaceGraph:
    def __init__(self, cfg):
        self.rooms = {k: tuple(v) for k, v in cfg.get("rooms", {}).items()}
        self.places = {}   # name -> dict(x, y, yaw(rad), room, aliases)
        self.nodes = {}    # every routable point: name -> (x, y, room)
        self.adj = {}      # name -> {neighbour: distance}
        self.start_place = cfg.get("start_place")
        for name, p in cfg.get("nodes", {}).items():
            self._add_node(name, p["x"], p["y"], p.get("room"))
        for name, p in cfg.get("places", {}).items():
            self.add_place(name, p["x"], p["y"], p.get("yaw"), p.get("room"),
                           p.get("aliases", []), connect=False)
        for a, b in cfg.get("edges", []):
            self.add_edge(a, b)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_files(cls, places_file, learned_file=None):
        with open(places_file) as f:
            cfg = yaml.safe_load(f)
        graph = cls(cfg)
        if learned_file and os.path.exists(learned_file):
            with open(learned_file) as f:
                learned = yaml.safe_load(f) or {}
            for name, p in learned.get("places", {}).items():
                graph.add_place(name, p["x"], p["y"], p.get("yaw"), None,
                                p.get("aliases", []), connect=True)
        return graph

    def _add_node(self, name, x, y, room=None):
        room = room or self.room_at(x, y)
        self.nodes[name] = (float(x), float(y), room)
        self.adj.setdefault(name, {})

    def add_edge(self, a, b):
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"edge {a}-{b}: unknown node")
        d = math.dist(self.nodes[a][:2], self.nodes[b][:2])
        self.adj[a][b] = d
        self.adj[b][a] = d

    def add_place(self, name, x, y, yaw_deg=None, room=None, aliases=(), connect=True):
        """Add a named destination. connect=True links it to the nearest node
        in the same room (used for places the robot learns at runtime)."""
        name = _norm(name).replace(" ", "_")
        room = room or self.room_at(x, y)
        nearest = self.nearest_node(x, y, same_room=True, room=room) if connect else None
        self._add_node(name, x, y, room)
        self.places[name] = {
            "x": float(x), "y": float(y),
            "yaw": None if yaw_deg is None else math.radians(yaw_deg),
            "room": room, "aliases": list(aliases),
        }
        if connect and nearest and nearest != name:
            self.add_edge(name, nearest)
        return name

    # ------------------------------------------------------------- queries
    def room_at(self, x, y):
        for name, (x0, y0, x1, y1) in self.rooms.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                # doorway band belongs to the hallway
                if name != "hallway" and "hallway" in self.rooms:
                    hx0, hy0, hx1, hy1 = self.rooms["hallway"]
                    if hx0 <= x <= hx1 and hy0 <= y <= hy1:
                        return "hallway"
                return name
        return None

    def nearest_node(self, x, y, same_room=False, room=None):
        room = room or self.room_at(x, y)
        candidates = [n for n, (_, _, r) in self.nodes.items() if not same_room or r == room]
        if not candidates:
            candidates = list(self.nodes)
        if not candidates:
            return None
        return min(candidates, key=lambda n: math.dist((x, y), self.nodes[n][:2]))

    def nearest_place(self, x, y):
        if not self.places:
            return None
        return min(self.places, key=lambda n: math.dist((x, y), self.nodes[n][:2]))

    def place_names(self):
        return sorted(self.places)

    def resolve(self, text):
        """Map free text ('the lounge', 'kitchn') to a canonical place name."""
        q = _norm(text)
        if not q:
            return None
        table = {}
        for name, p in self.places.items():
            table[_norm(name)] = name
            for a in p["aliases"]:
                table[_norm(a)] = name
        if q in table:
            return table[q]
        # longest known phrase contained in the text ("go to the kitchen please")
        hits = [k for k in table if re.search(rf"\b{re.escape(k)}\b", q)]
        if hits:
            return table[max(hits, key=len)]
        close = difflib.get_close_matches(q, table.keys(), n=1, cutoff=0.75)
        return table[close[0]] if close else None

    # ------------------------------------------------------------ planning
    def shortest_path(self, start, goal):
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist[u]:
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return path[::-1]

    def plan(self, x, y, place):
        """Return (node_names, [(x, y), ...], final_yaw) from pose to place."""
        if place not in self.places:
            raise KeyError(place)
        start = self.nearest_node(x, y, same_room=True)
        names = self.shortest_path(start, place)
        if names is None:
            return None
        pts = [self.nodes[n][:2] for n in names]
        # skip the first node if we are basically on it or it is behind us
        if len(pts) > 1 and math.dist((x, y), pts[0]) < 0.25:
            names, pts = names[1:], pts[1:]
        return names, pts, self.places[place]["yaw"]

    def path_length(self, names):
        return sum(self.adj[a][b] for a, b in zip(names, names[1:]))

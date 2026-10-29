"""Lidar person tracker for "follow me" (pure Python, no ROS).

Each scan is split into clusters (breaks where consecutive points jump by
more than `gap`). Person-sized clusters (legs, a body) are candidates; walls
and furniture are too wide. Nearby candidates are merged so two legs become
one person. The tracker locks onto a candidate and then, every scan, picks the
candidate closest to where the person should be (in the odom frame, so the
robot's own motion doesn't confuse it).
"""
import math


def scan_points(ranges, angle_min, angle_inc, rmin=0.12, rmax=8.0, offset=0.05):
    """Scan -> [(x, y)] in the robot base frame (lidar is offset forward)."""
    pts = []
    for i, r in enumerate(ranges):
        if rmin < r < rmax and math.isfinite(r):
            a = angle_min + i * angle_inc
            pts.append((offset + r * math.cos(a), r * math.sin(a)))
        else:
            pts.append(None)
    return pts


def clusters(pts, gap=0.18):
    """Split an ordered (wrapping) point list into clusters."""
    out, cur = [], []
    for p in pts:
        if p is None:
            if cur:
                out.append(cur)
            cur = []
            continue
        if cur and math.dist(cur[-1], p) > gap:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    # the scan wraps around: join last and first cluster if they touch
    if len(out) > 1 and pts and pts[0] is not None and pts[-1] is not None \
            and math.dist(out[0][0], out[-1][-1]) <= gap:
        out[0] = out[-1] + out[0]
        out.pop()
    return out


def candidates(pts, gap=0.18, min_w=0.05, max_w=0.75, merge=0.45, max_range=5.0):
    """Person-like blobs [(x, y, width)] in the base frame."""
    blobs = []
    for c in clusters(pts, gap):
        if len(c) < 2:
            continue
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        w = math.hypot(max(xs) - min(xs), max(ys) - min(ys))     # extent, robust to curved blobs
        if not (min_w <= w <= max_w):
            continue
        cx = sum(p[0] for p in c) / len(c)
        cy = sum(p[1] for p in c) / len(c)
        if math.hypot(cx, cy) <= max_range:
            blobs.append([cx, cy, w, 1])
    # merge legs: blobs whose centres are close become one person
    merged = []
    for b in blobs:
        for m in merged:
            if math.dist(b[:2], m[:2]) < merge:
                n = m[3] + 1
                m[0], m[1] = (m[0] * m[3] + b[0]) / n, (m[1] * m[3] + b[1]) / n
                m[2], m[3] = max(m[2], b[2]), n
                break
        else:
            merged.append(list(b))
    return [(m[0], m[1], m[2]) for m in merged]


def base_to_odom(pose, p):
    x, y, yaw = pose
    return (x + p[0] * math.cos(yaw) - p[1] * math.sin(yaw),
            y + p[0] * math.sin(yaw) + p[1] * math.cos(yaw))


def odom_to_base(pose, p):
    x, y, yaw = pose
    dx, dy = p[0] - x, p[1] - y
    return (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))


class PersonTracker:
    def __init__(self, gate=0.7, lost_after=3.0, acquire_range=3.5, acquire_fov=math.radians(360)):
        self.gate, self.lost_after = gate, lost_after
        self.acquire_range, self.acquire_fov = acquire_range, acquire_fov
        self.target = None          # (x, y) in odom
        self.vel = (0.0, 0.0)
        self.last_seen = None

    def acquire(self, cands, pose, prefer_bearing=None):
        """Lock onto the nearest candidate (or the one nearest prefer_bearing)."""
        ok = [c for c in cands if math.hypot(c[0], c[1]) <= self.acquire_range
              and abs(math.atan2(c[1], c[0])) <= self.acquire_fov / 2]
        if not ok:
            return False
        if prefer_bearing is not None:
            pick = min(ok, key=lambda c: abs(math.atan2(c[1], c[0]) - prefer_bearing))
        else:
            pick = min(ok, key=lambda c: math.hypot(c[0], c[1]))
        self.target = base_to_odom(pose, pick)
        self.vel = (0.0, 0.0)
        return True

    def update(self, cands, pose, t, dt=0.1):
        """Associate this scan's candidates with the target. Returns the target
        in the base frame (x, y) or None if not seen this scan."""
        if self.target is None:
            return None
        pred = (self.target[0] + self.vel[0] * dt, self.target[1] + self.vel[1] * dt)
        best, bd = None, self.gate
        for c in cands:
            w = base_to_odom(pose, c)
            d = math.dist(w, pred)
            if d < bd:
                best, bd = w, d
        if best is None:
            return None
        if self.last_seen is not None and t > self.last_seen:
            k = 0.4
            el = max(t - self.last_seen, 1e-3)
            vx, vy = (best[0] - self.target[0]) / el, (best[1] - self.target[1]) / el
            self.vel = ((1 - k) * self.vel[0] + k * vx, (1 - k) * self.vel[1] + k * vy)
        self.target, self.last_seen = best, t
        return odom_to_base(pose, best)

    def lost(self, t):
        return self.last_seen is not None and t - self.last_seen > self.lost_after


def follow_command(target, distance=0.9, vmax=0.5, wmax=1.2, front_clear=math.inf):
    """P-controller: keep `distance` behind the target, face it.
    target: (x, y) in the base frame. Returns (v, w)."""
    tx, ty = target
    d = math.hypot(tx, ty)
    bearing = math.atan2(ty, tx)
    w = max(-wmax, min(wmax, 2.0 * bearing))
    err = d - distance
    v = 0.0
    if abs(bearing) < 0.9:
        v = max(-0.15, min(vmax, 0.9 * err)) * math.cos(bearing)
        if abs(err) < 0.08:
            v = 0.0
    if v > 0 and front_clear < 0.35:
        v = 0.0
    return v, w

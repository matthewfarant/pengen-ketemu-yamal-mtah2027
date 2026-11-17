"""Read the Sportec Open DFL dataset (Bassek et al. 2025, CC BY 4.0) into a Match.

External training data: 7 Bundesliga / 2. Bundesliga games, 25 Hz positions with a per-frame
BallPossession (1 home, 2 guest) and BallStatus (1 alive, 0 dead) on the ball frames. Those two
flags are exactly the label space of the challenge (H / A / D).
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

from .common import Match

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(.*)/>')
POSS = re.compile(r'BallPossession="(\d)"')
STATUS = re.compile(r'BallStatus="(\d)"')


def match_info(path):
    root = ET.parse(path).getroot()
    env = root.find(".//Environment")
    length, width = float(env.get("PitchX")), float(env.get("PitchY"))
    teams = {}
    gks = set()
    for t in root.iter("Team"):
        teams[t.get("TeamId")] = 0 if t.get("Role") == "home" else 1
        for p in t.iter("Player"):
            if p.get("PlayingPosition") == "TW":
                gks.add(p.get("PersonId"))
    return length, width, teams, gks


def load(info_path, pos_path, name=""):
    length, width, teams, gks = match_info(info_path)
    data = {}  # (half, person) -> list of (N, x, y)
    ball = {}  # half -> list of (N, x, y, poss, status)
    person_team = {}
    cur = None
    with open(pos_path) as fh:
        for line in fh:
            m = FRAME.search(line)
            if m:
                if cur is None:
                    continue
                n, x, y, rest = int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
                if cur[1] == "BALL":
                    p = POSS.search(rest)
                    s = STATUS.search(rest)
                    ball[cur[0]].append((n, x, y, int(p.group(1)) if p else 0,
                                         int(s.group(1)) if s else 0))
                else:
                    data[cur].append((n, x, y))
                continue
            m = FRAMESET.search(line)
            if m:
                half, team_id, person = m.groups()
                if team_id == "referee":
                    cur = None
                elif team_id == "BALL":
                    cur = (half, "BALL")
                    ball.setdefault(half, [])
                else:
                    cur = (half, person)
                    person_team[person] = teams[team_id]
                    data.setdefault(cur, [])
    persons = sorted(person_team, key=lambda p: (person_team[p], p))
    slot = {p: i for i, p in enumerate(persons)}
    halves = [h for h in ("firstHalf", "secondHalf") if h in ball]
    blocks = []
    for hi, h in enumerate(halves):
        b = np.array(ball[h], dtype=np.float64)
        n0 = int(b[:, 0].min())
        T = int(b[:, 0].max()) - n0 + 1
        bx = np.full(T, np.nan); by = np.full(T, np.nan)
        lab = np.full(T, 3, np.int8)
        idx = b[:, 0].astype(int) - n0
        bx[idx] = b[:, 1]; by[idx] = b[:, 2]
        poss, stat = b[:, 3].astype(int), b[:, 4].astype(int)
        l = np.where(stat == 0, 2, np.where(poss == 1, 0, np.where(poss == 2, 1, 3)))
        lab[idx] = l
        px = np.full((T, len(persons)), np.nan); py = np.full((T, len(persons)), np.nan)
        for (hh, person), rows in data.items():
            if hh != h or not rows:
                continue
            r = np.array(rows)
            k = r[:, 0].astype(int) - n0
            ok = (k >= 0) & (k < T)
            px[k[ok], slot[person]] = r[ok, 1]
            py[k[ok], slot[person]] = r[ok, 2]
        blocks.append((px, py, bx, by, lab, np.full(T, hi + 1, np.int8)))
    px, py, bx, by, lab, half = (np.concatenate(z) for z in zip(*blocks))
    team = np.array([person_team[p] for p in persons], np.int8)
    gk = np.array([p in gks for p in persons])
    # DFL origin is the centre spot; move it to a corner like the ATD
    return Match(px + length / 2, py + width / 2, team, gk, bx + length / 2, by + width / 2,
                 length, width, half=half, labels=lab, name=name)

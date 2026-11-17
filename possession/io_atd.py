"""Read the provided Metrica/EPTS ATD (tactical or broadcast) into a Match."""
import xml.etree.ElementTree as ET

import numpy as np

from .common import Match

PITCH = (105.3, 68.0)  # games.GAMES["BAR-ATH"]["pitch"]


def slot_teams(meta_path):
    """Team (0 home = first team in the XML, 1 away) and goalkeeper flag per track slot.

    The ATD slots are tracklets, not people. The goalkeeper is the last slot of each team, which is
    where the provider puts the keeper track (its speed column is NaN throughout)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team = np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)
    gk = np.zeros(len(team), bool)
    for k in (0, 1):
        gk[np.where(team == k)[0][-1]] = True
    return team, gk


def load(meta_path, raw_path, n_frames, pitch=PITCH):
    team, gk = slot_teams(meta_path)
    n = len(team)
    px = np.full((n_frames, n), np.nan, np.float32)
    py = np.full((n_frames, n), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            fid, players, ball = line.rstrip("\n").split(":")
            i = int(fid) - 1  # ATD frame_id counts from 1, video frames from 0
            if i < 0 or i >= n_frames:
                continue
            for s, g in enumerate(players.rstrip(";").split(";")):
                if g[0] != "N":
                    a, b, _ = g.split(",", 2)
                    px[i, s] = float(a)
                    py[i, s] = float(b)
            if ball[0] != "N":
                a, b, _ = ball.split(",", 2)
                bx[i] = float(a)
                by[i] = float(b)
    L, W = pitch
    return Match(px * L, py * W, team, gk, bx * L, by * W, L, W, name="BAR-ATH")

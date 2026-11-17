"""Shared match representation.

Every source (the provided ATD, the external DFL data) is turned into the same arrays:

    px, py   float32 (T, N)  player positions in metres, origin at a pitch corner, NaN = not tracked
    team     int8    (N,)    0 = home, 1 = away
    gk       bool    (N,)    slot is a goalkeeper
    bx, by   float32 (T,)    ball position in metres, NaN = ball not tracked
    length, width            pitch size in metres
    fps                      25
"""
import numpy as np

FPS = 25


class Match:
    def __init__(self, px, py, team, gk, bx, by, length, width, half=None, labels=None, name=""):
        self.px, self.py = px.astype(np.float32), py.astype(np.float32)
        self.team = np.asarray(team, dtype=np.int8)
        self.gk = np.asarray(gk, dtype=bool)
        self.bx, self.by = bx.astype(np.float32), by.astype(np.float32)
        self.length, self.width = float(length), float(width)
        # half index per frame (1, 2) or 0 outside play; labels: int8 0=H 1=A 2=D 3=X
        self.half = half
        self.labels = labels
        self.name = name

    @property
    def n(self):
        return len(self.bx)

    def save(self, path):
        np.savez_compressed(path, px=self.px, py=self.py, team=self.team, gk=self.gk, bx=self.bx,
                            by=self.by, pitch=np.array([self.length, self.width]),
                            half=self.half if self.half is not None else np.zeros(0, np.int8),
                            labels=self.labels if self.labels is not None else np.zeros(0, np.int8),
                            name=np.array(self.name))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        half = z["half"] if len(z["half"]) else None
        labels = z["labels"] if len(z["labels"]) else None
        return cls(z["px"], z["py"], z["team"], z["gk"], z["bx"], z["by"], *z["pitch"],
                   half=half, labels=labels, name=str(z["name"]))

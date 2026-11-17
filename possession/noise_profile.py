"""Measure how incomplete the provided tracking is, so the clean external data can be degraded to
match it. Uses only the provided tracking (no labels, no video): gap/run lengths of the ball track,
where on the pitch the ball tends to vanish, and how fragmented the player tracks are."""
import json

import numpy as np


def runs(mask):
    """(starts, lengths) of True runs in a boolean array."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return s, e - s


ZX = np.array([0, 16.5, 35, 52.65, 70.3, 88.8, 105.3]) / 105.3  # fractions of length
ZY = np.array([0, 13.84, 54.16, 68]) / 68


def zone_of(x, y, length, width):
    """Coarse zone id: 6 bands along the pitch (boxes, thirds) x 3 across (wings / centre), plus
    18 = outside the pitch."""
    xf, yf = x / length, y / width
    out = (xf < 0) | (xf > 1) | (yf < 0) | (yf > 1)
    ix = np.clip(np.searchsorted(ZX, xf, side="right") - 1, 0, 5)
    iy = np.clip(np.searchsorted(ZY, yf, side="right") - 1, 0, 2)
    z = ix * 3 + iy
    z[out] = 18
    return z


def measure(match, play_mask):
    bx, by = match.bx, match.by
    present = np.isfinite(bx)
    # ball gaps / present runs inside play only
    s, l = runs(~present & play_mask)
    gaps = l[l >= 1]
    s2, l2 = runs(present & play_mask)
    # zone where the ball was last seen before each gap -> relative hazard of losing it there
    zone_all = np.full(len(bx), -1)
    zone_all[present] = zone_of(bx[present], by[present], match.length, match.width)
    starts_zone = np.array([zone_all[i - 1] for i in s if i > 0 and present[i - 1]])
    pres_zone = zone_all[present & play_mask]
    exposure = np.bincount(pres_zone, minlength=19).astype(float)
    events = np.bincount(starts_zone, minlength=19).astype(float)
    hazard = (events + 1) / (exposure + 50)
    # players
    tracked = np.isfinite(match.px)
    per_frame = tracked[play_mask].sum(1)
    frag = []
    holes = []
    for k in range(tracked.shape[1]):
        col = tracked[:, k] & play_mask
        _, fl = runs(col)
        frag.extend(fl.tolist())
    return {
        "ball_missing_frac": float(1 - present[play_mask].mean()),
        "ball_gap_lengths": gaps.astype(int).tolist(),
        "ball_run_lengths": l2.astype(int).tolist(),
        "ball_gap_hazard_by_zone": hazard.tolist(),
        "players_per_frame_mean": float(per_frame.mean()),
        "players_per_frame_hist": np.bincount(per_frame, minlength=30).tolist(),
        "player_fragment_lengths": [int(x) for x in frag],
    }


def save(profile, path):
    with open(path, "w") as fh:
        json.dump(profile, fh)

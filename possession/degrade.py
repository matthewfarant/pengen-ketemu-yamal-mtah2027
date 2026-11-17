"""Degrade clean external tracking to look like the provided automatic tracking.

The external data (DFL) is corrected optical tracking: ball always there, 22 players always there,
centimetre accuracy. Ours loses the ball for long stretches (41% of frames, gaps of 6-50 s) and
drops or fragments player tracks. A model trained on clean data would never learn to work through
the gaps, so every external match is degraded several times with different seeds, using gap and
fragment statistics measured on the provided tracking (models/noise_profile.json).
"""
import os

import numpy as np

from .common import Match
from .noise_profile import runs, zone_of


def _outside_mask(match, keep_frames):
    """The provided tracking almost never follows the ball once it leaves the pitch (3.7% of
    visible-ball frames are outside, against 25% in the external data, where the ball is tracked
    into the hands of the ball boys). Keep only the first frames of every excursion."""
    edge = np.minimum.reduce([match.bx, match.length - match.bx, match.by, match.width - match.by])
    out = edge < -0.3
    keep = np.ones(match.n, bool)
    s, l = runs(out)
    for a, n in zip(s, l):
        keep[a + keep_frames:a + n] = False
    return keep


def _gap_process(p_start, gaps, u, pick):
    n = len(p_start)
    keep = np.ones(n, bool)
    i = j = 0
    while i < n:
        if u[i] < p_start[i]:
            g = int(gaps[pick[j % len(pick)]])
            j += 1
            keep[i:i + g] = False
            i += g
        else:
            i += 1
    return keep


def _ball_mask(match, profile, rng, keep_outside=12):
    """True where the ball stays visible. First the off-pitch excursions are cut, then alternating
    visible/invisible runs: a gap starts with a zone-dependent hazard and lasts a length drawn from
    the measured gap lengths. The gap rate is scaled (3 fixed-point steps, same random draws) until
    the missing share matches the provided tracking."""
    n = match.n
    keep0 = _outside_mask(match, keep_outside)
    gaps = np.asarray(profile["ball_gap_lengths"])
    hz = np.asarray(profile["ball_gap_hazard_by_zone"])
    target = profile["ball_missing_frac"]
    z = zone_of(match.bx, match.by, match.length, match.width)
    base = hz[z] / hz[z].mean()
    u = rng.random(n)
    pick = rng.integers(len(gaps), size=n // 50 + 10)
    rate = target / ((1 - target) * gaps.mean())
    for _ in range(4):
        keep = _gap_process(np.clip(base * rate, 0, 0.5), gaps, u, pick) & keep0
        got = 1 - keep.mean()
        rate *= (target / (1 - target)) / max(got / (1 - got), 1e-6)
    return keep


def _player_mask(match, profile, rng):
    """Fragment each player track: visible runs drawn from the measured fragment lengths,
    separated by short holes, tuned to the measured mean players per frame."""
    n, k = match.px.shape
    frags = np.asarray(profile["player_fragment_lengths"])
    present = np.isfinite(match.px)
    target = profile["players_per_frame_mean"]
    clean = present.sum(1).mean()
    miss = float(np.clip(1 - target / max(clean, 1e-6), 0.0, 0.5))
    mean_frag = frags.mean()
    mean_hole = mean_frag * miss / max(1 - miss, 1e-6)
    keep = np.ones((n, k), bool)
    for j in range(k):
        i = int(rng.integers(0, int(mean_frag)))
        while i < n:
            h = int(rng.exponential(mean_hole)) if mean_hole > 0 else 0
            keep[i:i + h, j] = False
            i += h + int(frags[rng.integers(len(frags))])
    return keep


def _ar_noise(n, sigma, rho, rng, shape=()):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), (n,) + shape)
    out = np.empty_like(e)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


BALL_SMOOTH = int(os.environ.get("MTAH_BALL_SMOOTH", "5"))


def _smooth(x, w):
    k = np.ones(w) / w
    pad = np.pad(x.astype(np.float64), w // 2, mode="edge")
    return np.convolve(pad, k, mode="valid").astype(np.float32)


def _slow_noise(n, sigma, rng, step=25):
    knots = rng.normal(0, sigma, n // step + 2)
    return np.interp(np.arange(n), np.arange(len(knots)) * step, knots).astype(np.float32)


def degrade(match, profile, seed, pos_sigma=0.0, ball_sigma=1.0):
    rng = np.random.default_rng(seed)
    n, k = match.px.shape
    keep_b = _ball_mask(match, profile, rng)
    keep_p = _player_mask(match, profile, rng)
    # player positions: the provided tracking is already smoother than the external data, so no
    # extra noise by default
    px = match.px + _ar_noise(n, pos_sigma, 0.95, rng, (k,)).astype(np.float32) if pos_sigma else match.px.copy()
    py = match.py + _ar_noise(n, pos_sigma, 0.95, rng, (k,)).astype(np.float32) if pos_sigma else match.py.copy()
    # the provided ball track is smoother than the external one (moving average whose width matches
    # the 99th percentile of velocity change, 7.6 m/s vs 7.4 m/s) and offset by up
    # to a couple of metres from the real ball (a slowly varying error, knots once a second)
    bx = _smooth(match.bx, BALL_SMOOTH) + _slow_noise(n, ball_sigma, rng)
    by = _smooth(match.by, BALL_SMOOTH) + _slow_noise(n, ball_sigma, rng)
    px[~keep_p] = np.nan
    py[~keep_p] = np.nan
    bx[~keep_b] = np.nan
    by[~keep_b] = np.nan
    return Match(px, py, match.team, match.gk, bx, by, match.length, match.width,
                 half=match.half, labels=match.labels, name=f"{match.name}#s{seed}")

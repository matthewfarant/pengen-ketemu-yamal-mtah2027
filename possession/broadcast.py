"""Fill ball gaps in the tactical tracking with the broadcast tracking.

The two recordings run on separate clocks. Matching their ball tracks window by window gives one
constant offset per half (tactical = broadcast + offset), found by search, not by hand. Replays
and other non-live broadcast footage are rejected frame by frame: a broadcast frame counts as live
only if its visible players sit on top of the tactical players at the mapped frame.
"""
import numpy as np


def window_offsets(t, b, guesses, W=125, search=150, min_common=40):
    """[(broadcast_start, offset, median ball distance)] for windows where both balls are seen."""
    res = []
    for f0 in range(0, b.n - W, W):
        bb = np.c_[b.bx[f0:f0 + W], b.by[f0:f0 + W]]
        okb = np.isfinite(bb[:, 0])
        if okb.sum() < 50:
            continue
        best = None
        for g in guesses:
            for o in range(g - search, g + search + 1):
                a = f0 + o
                if a < 0 or a + W > t.n:
                    continue
                tt = np.c_[t.bx[a:a + W], t.by[a:a + W]]
                ok = okb & np.isfinite(tt[:, 0])
                if ok.sum() < min_common:
                    continue
                d = np.median(np.hypot(*(bb[ok] - tt[ok]).T))
                if best is None or d < best[2]:
                    best = (f0, o, d)
        if best is not None:
            res.append(best)
    return np.array(res)


def offsets_by_frame(windows, n_b, max_dist=1.5):
    """Piecewise-constant offset per broadcast frame: each window's well-matched offset, with the
    most common offset among neighbours in time (median filter) to remove outliers, held between
    windows (nearest window)."""
    w = windows[windows[:, 2] < max_dist]
    w = w[np.argsort(w[:, 0])]
    off = w[:, 1].copy()
    k = 7
    sm = np.array([np.median(off[max(0, i - k):i + k + 1]) for i in range(len(off))])
    centres = w[:, 0] + 62
    idx = np.clip(np.searchsorted(centres, np.arange(n_b)), 0, len(centres) - 1)
    prev = np.clip(idx - 1, 0, len(centres) - 1)
    choose_prev = np.abs(np.arange(n_b) - centres[prev]) < np.abs(np.arange(n_b) - centres[idx])
    j = np.where(choose_prev, prev, idx)
    return np.round(sm[j]).astype(int)


def live_mask(t, b, off, thresh=1.5, min_players=4, chunk=5000):
    """Per broadcast frame: True if the median distance from each visible broadcast player to the
    nearest tactical player at the mapped frame is below thresh metres."""
    n = b.n
    live = np.zeros(n, bool)
    score = np.full(n, np.nan, np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        bf = np.arange(s, e)
        tf = bf + off[s:e]
        ok = (tf >= 0) & (tf < t.n)
        bf, tf = bf[ok], tf[ok]
        bpx, bpy = b.px[bf], b.py[bf]
        tpx, tpy = t.px[tf], t.py[tf]
        d = np.hypot(bpx[:, :, None] - tpx[:, None, :], bpy[:, :, None] - tpy[:, None, :])
        d = np.where(np.isfinite(d), d, np.inf).min(2)
        d[~np.isfinite(d)] = np.nan
        cnt = np.isfinite(d).sum(1)
        with np.errstate(all="ignore"):
            med = np.nanmedian(d, 1)
        score[bf] = med
        live[bf] = (cnt >= min_players) & (med < thresh)
    return live, score


def fill_ball(t, b, off, live):
    """Copy of the tactical ball with gaps filled from live broadcast frames. Returns
    (bx, by, source) with source 0 = tactical, 1 = broadcast, -1 = missing."""
    bx, by = t.bx.copy(), t.by.copy()
    src = np.where(np.isfinite(bx), 0, -1).astype(np.int8)
    bf = np.arange(b.n)
    tf = bf + off
    ok = live & np.isfinite(b.bx) & (tf >= 0) & (tf < t.n)
    bf, tf = bf[ok], tf[ok]
    gap = ~np.isfinite(bx[tf])
    bx[tf[gap]] = b.bx[bf[gap]]
    by[tf[gap]] = b.by[bf[gap]]
    src[tf[gap]] = 1
    return bx, by, src

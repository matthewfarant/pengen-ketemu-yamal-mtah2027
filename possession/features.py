"""Per-frame features, from the point of view of one team ("us") against the other ("them").

Everything is oriented so that "us" attacks towards +x, which makes the features identical in
meaning for home and away and for both halves. The model is trained on both points of view of every
external match, and at prediction time both points of view are averaged, so it carries no notion
of which team is home.

The labels are offline, so context windows look both backwards and forwards in time.
"""
import os

import numpy as np

from .common import FPS

# feature-set switch (CV experiments); the submitted configuration is the default
USE_TOUCH = os.environ.get("MTAH_TOUCH", "0") == "1"

BOX_LEN, BOX_W = 16.5, 40.32
FAST = 4.0  # m/s, "running"


def attack_sign(match):
    """Per frame and team: +1 if the team attacks towards +x. Decided per half from where each
    team's goalkeeper (or, failing that, its centroid) stands on average."""
    half = match.half if match.half is not None else np.ones(match.n, np.int8)
    sign = np.ones((match.n, 2), np.int8)
    mid = match.length / 2
    for h in np.unique(half):
        if h == 0:
            continue
        rows = half == h
        for k in (0, 1):
            cols = (match.team == k) & match.gk
            x = match.px[rows][:, cols]
            if not np.isfinite(x).any():
                x = match.px[rows][:, match.team == k]
            sign[rows, k] = 1 if np.nanmean(x) < mid else -1
    return sign


def _velocity(p, step=2):
    v = np.full_like(p, np.nan)
    v[step:-step] = (p[2 * step:] - p[:-2 * step]) * (FPS / (2 * step))
    return v


def _roll(x, w, offset=0):
    """NaN-aware moving mean over a window of w frames centred at t+offset."""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    c = np.r_[0, np.cumsum(np.where(ok, x, 0))]
    n = np.r_[0, np.cumsum(ok)]
    T = len(x)
    lo = np.clip(np.arange(T) + offset - w // 2, 0, T)
    hi = np.clip(np.arange(T) + offset + (w - w // 2), 0, T)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan).astype(np.float32)


def _since(ok, cap=3000):
    """Frames since the last True (0 on True frames), capped."""
    T = len(ok)
    idx = np.where(ok, np.arange(T), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(T) - last, cap).astype(np.float32)


def _carry(x, ok):
    """Last value of x on an ok frame, carried forward (NaN before the first)."""
    T = len(x)
    idx = np.where(ok, np.arange(T), 0)
    idx = np.maximum.accumulate(idx)
    out = x[idx].astype(np.float32)
    out[~np.maximum.accumulate(ok)] = np.nan
    return out


def _carry_back(x, ok):
    return _carry(x[::-1], ok[::-1])[::-1]


def touches(bx, by, d_near, team_near, jump=4.0, reach=2.5):
    """Ball touches: frames where the ball's velocity changes sharply (a kick, a stop, a deflection)
    with a player close enough to have caused it. Returns per-frame touching team (0/1) or -1."""
    vx, vy = _velocity(bx.astype(np.float64), 2), _velocity(by.astype(np.float64), 2)
    dv = np.full(len(bx), np.nan)
    dv[3:-3] = np.hypot(vx[6:] - vx[:-6], vy[6:] - vy[:-6])
    ok = np.isfinite(dv)
    peak = np.zeros(len(bx), bool)
    peak[1:-1] = ok[1:-1] & (dv[1:-1] > jump) & (dv[1:-1] >= np.nan_to_num(dv[:-2], nan=0)) & \
        (dv[1:-1] > np.nan_to_num(dv[2:], nan=0))
    team = np.full(len(bx), -1, np.int8)
    sel = peak & (d_near < reach)
    team[sel] = team_near[sel]
    return team


def base_features(match):
    """Per-frame, perspective-independent arrays reused for both points of view."""
    L, W = match.length, match.width
    px, py, bx, by = match.px, match.py, match.bx, match.by
    vx, vy = _velocity(px), _velocity(py)
    spd = np.hypot(vx, vy)
    bad = spd > 12
    vx[bad] = np.nan; vy[bad] = np.nan; spd[bad] = np.nan
    bvx, bvy = _velocity(bx), _velocity(by)
    bspd = np.hypot(bvx, bvy)
    bspd[bspd > 45] = np.nan
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dd = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(dd, 1)
    d_near = dd[np.arange(len(bx)), j]
    touch = touches(bx, by, d_near, match.team[j])
    return dict(touch=touch, L=L, W=W, px=px, py=py, vx=vx, vy=vy, spd=spd, bx=bx, by=by, bspd=bspd,
                bvx=bvx, bvy=bvy, d=d, sign=attack_sign(match))


def _touch_features(F, base, us, n):
    """Who played the ball last / next, and how many touches in a row by the same team."""
    tch = base["touch"]
    is_t = tch >= 0
    t_us = np.where(is_t, (tch == us).astype(np.float32), np.nan)
    F["touch_last_us"] = _carry(t_us, is_t)
    F["touch_next_us"] = _carry_back(t_us, is_t)
    F["touch_since"] = _since(is_t, 750)
    F["touch_until"] = _since(is_t[::-1], 750)[::-1].copy()
    for w in (75, 250):
        F[f"touch_us_p{w}"] = _roll(t_us, w, -w // 2)
        F[f"touch_us_f{w}"] = _roll(t_us, w, w // 2)
        F[f"touch_n_p{w}"] = _roll(is_t.astype(np.float32), w, -w // 2) * w
    # chain: consecutive same-team touches ending at the last touch / starting at the next one
    idx = np.flatnonzero(is_t)
    chain = np.zeros(len(idx), np.float32)
    for i in range(1, len(idx)):
        chain[i] = chain[i - 1] + 1 if tch[idx[i]] == tch[idx[i - 1]] else 0
    chain_b = np.zeros(len(idx), np.float32)
    for i in range(len(idx) - 2, -1, -1):
        chain_b[i] = chain_b[i + 1] + 1 if tch[idx[i]] == tch[idx[i + 1]] else 0
    cf = np.full(n, np.nan, np.float32); cf[idx] = chain
    cb = np.full(n, np.nan, np.float32); cb[idx] = chain_b
    F["chain_last"] = _carry(cf, is_t)
    F["chain_next"] = _carry_back(cb, is_t)


def features(match, base, us):
    L, W = base["L"], base["W"]
    them = 1 - us
    s = base["sign"][:, us].astype(np.float32)  # +1: us attacks +x
    team = match.team
    gk = match.gk
    px, py, vx, spd, d = base["px"], base["py"], base["vx"], base["spd"], base["d"]
    xo = np.where(s[:, None] > 0, px, L - px)  # oriented x: us attacks +x
    vxo = vx * s[:, None]
    bx, by = base["bx"], base["by"]
    bxo = np.where(s > 0, bx, L - bx)
    b_ok = np.isfinite(bx)
    F = {}
    F["b_ok"] = b_ok.astype(np.float32)
    F["bxo"] = bxo
    F["byc"] = np.abs(by - W / 2)
    edge = np.minimum.reduce([bx, L - bx, by, W - by])
    F["b_edge"] = edge
    F["b_out"] = np.where(b_ok, (edge < -0.3).astype(np.float32), np.nan)
    F["bspd"] = base["bspd"]
    F["bspd_s"] = _roll(base["bspd"], 9)
    F["bacc"] = np.abs(_roll(base["bspd"], 3, 2) - _roll(base["bspd"], 3, -2))
    tu, tt = team == us, team == them
    out_tu, out_tt = tu & ~gk, tt & ~gk
    inf = np.float32(np.inf)
    dd = np.where(np.isfinite(d), d, inf)
    du = dd[:, tu].min(1); dt = dd[:, tt].min(1)
    du[~np.isfinite(du)] = np.nan; dt[~np.isfinite(dt)] = np.nan
    F["d_us"], F["d_them"] = du, dt
    F["d_diff"] = du - dt
    F["d_gk_us"] = np.where(np.isfinite(d[:, tu & gk]).any(1), np.nanmin(np.where(np.isfinite(d[:, tu & gk]), d[:, tu & gk], 1e3), 1), np.nan)
    F["d_gk_them"] = np.where(np.isfinite(d[:, tt & gk]).any(1), np.nanmin(np.where(np.isfinite(d[:, tt & gk]), d[:, tt & gk], 1e3), 1), np.nan)
    for r in (2, 5):
        F[f"n_us_{r}"] = np.where(b_ok, (dd[:, tu] < r).sum(1), np.nan).astype(np.float32)
        F[f"n_them_{r}"] = np.where(b_ok, (dd[:, tt] < r).sum(1), np.nan).astype(np.float32)
    F["n_diff_2"] = F["n_us_2"] - F["n_them_2"]
    # ball moving with its nearest player = carrying it
    j = np.argmin(dd, 1)
    rows = np.arange(match.n)
    nvx = base["vx"][rows, j]; nvy = base["vy"][rows, j]
    F["b_rel_v"] = np.hypot(base["bvx"] - nvx, base["bvy"] - nvy)
    F["near_is_us"] = np.where(b_ok & np.isfinite(dd[rows, j]), (team[j] == us).astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        for nm, cols, ocols in (("us", tu, out_tu), ("them", tt, out_tt)):
            X = xo[:, ocols]; Y = py[:, ocols]
            F[f"cnt_{nm}"] = np.isfinite(px[:, cols]).sum(1).astype(np.float32)
            F[f"cx_{nm}"] = np.nanmean(X, 1)
            F[f"sx_{nm}"] = np.nanstd(X, 1)
            F[f"sy_{nm}"] = np.nanstd(Y, 1)
            F[f"xmax_{nm}"] = np.nanmax(X, 1)
            F[f"xmin_{nm}"] = np.nanmin(X, 1)
            F[f"vx_{nm}"] = np.nanmean(vxo[:, ocols], 1)
            F[f"spd_{nm}"] = np.nanmean(spd[:, ocols], 1)
            F[f"fast_{nm}"] = (spd[:, ocols] > FAST).sum(1).astype(np.float32)
            F[f"box_att_{nm}"] = ((X > L - BOX_LEN) & (np.abs(Y - W / 2) < BOX_W / 2)).sum(1).astype(np.float32)
            F[f"box_def_{nm}"] = ((X < BOX_LEN) & (np.abs(Y - W / 2) < BOX_W / 2)).sum(1).astype(np.float32)
            F[f"ownhalf_{nm}"] = (X < L / 2).sum(1).astype(np.float32)
        F["spd_all"] = np.nanmean(spd[:, ~gk], 1)
        F["cx_diff"] = F["cx_us"] - F["cx_them"]
        F["sy_diff"] = F["sy_us"] - F["sy_them"]
        F["sx_diff"] = F["sx_us"] - F["sx_them"]
        F["vx_diff"] = F["vx_us"] - F["vx_them"]
        F["b_vs_line"] = bxo - F["xmax_them"]
        F["b_vs_cx"] = bxo - F["cx_us"]
    if USE_TOUCH:
        _touch_features(F, base, us, match.n)
    # context windows, both directions
    ctx = ["b_ok", "d_diff", "n_diff_2", "near_is_us", "bspd", "spd_all", "vx_us", "vx_them",
           "cx_diff", "sy_diff", "b_out", "b_edge", "bxo", "fast_us", "fast_them", "b_rel_v"]
    for k in ctx:
        x = F[k]
        for w in (5, 25, 75, 250):
            F[f"{k}_c{w}"] = _roll(x, w)
        F[f"{k}_p50"] = _roll(x, 50, -25)
        F[f"{k}_f50"] = _roll(x, 50, 25)
    for k in ("spd_all", "near_is_us", "d_diff"):
        F[f"{k}_c750"] = _roll(F[k], 750)
    # bridging the ball gaps: what we knew when it vanished and when it came back
    F["since_ball"] = _since(b_ok)
    F["until_ball"] = _since(b_ok[::-1])[::-1].copy()
    for k in ("bxo", "b_edge", "near_is_us_c25", "d_diff_c25", "b_out"):
        F[f"{k}_last"] = _carry(F[k], b_ok)
        F[f"{k}_next"] = _carry_back(F[k], b_ok)
    names = sorted(F)
    X = np.stack([np.asarray(F[k], np.float32) for k in names], 1)
    return X, names


def labels_for(match, us):
    """0 = us, 1 = them, 2 = dead, 3 = unknown, from H/A/D/X codes."""
    lab = match.labels
    out = np.full(len(lab), 3, np.int8)
    out[lab == 2] = 2
    out[lab == us] = 0
    out[lab == 1 - us] = 1
    return out

"""Bring the external possession flags in line with the challenge's definition of possession.

DFL's BallPossession flips on touches: about 5 team spells per minute, many under a second. The
challenge counts a change only on *controlled* possession: "a simple touch of the ball without
control", "a deflection that does not intercept" and "an aerial duel won without subsequent
control" do not change possession, and loose balls stay with the last team (CHALLENGE.md). So a
team spell shorter than MIN_CONTROL frames that does not start a restart is treated as an
uncontrolled touch and handed back to the team that had the ball before it.
"""
import numpy as np

H, A, D, X = 0, 1, 2, 3
MIN_CONTROL = 25  # 1 s at 25 fps: shorter than a receive-and-play, i.e. a touch without control


def segments(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(lab)]
    return st, en, lab[st]


def apply_control_rule(lab, half=None, min_control=MIN_CONTROL):
    out = lab.copy()
    blocks = [np.arange(len(lab))] if half is None else [np.flatnonzero(half == h) for h in np.unique(half)]
    for idx in blocks:
        seg = out[idx]
        changed = True
        while changed:
            changed = False
            st, en, v = segments(seg)
            for i in range(1, len(v)):
                if v[i] in (H, A) and en[i] - st[i] < min_control and v[i - 1] in (H, A) \
                        and v[i - 1] != v[i]:
                    nxt = v[i + 1] if i + 1 < len(v) else D
                    if nxt in (v[i - 1], D):
                        seg[st[i]:en[i]] = v[i - 1]
                        changed = True
                        break
        out[idx] = seg
    return out

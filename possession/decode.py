"""Turn per-frame probabilities into possession spells.

A possession is an interval, and the ranking metric punishes a one-frame flicker as hard as a
missed possession, so the per-frame probabilities go through a Viterbi decoder over the three
states with a cost for every change of state. The costs are tuned on the external matches only.
"""
import numpy as np


def viterbi(logp, c_switch, c_to_dead, c_from_dead):
    """logp: (T, 3) log-probabilities for H, A, D. Returns int8 codes 0/1/2."""
    T = len(logp)
    trans = np.array([[0, c_switch, c_to_dead],
                      [c_switch, 0, c_to_dead],
                      [c_from_dead, c_from_dead, 0]], dtype=np.float64)
    back = np.zeros((T, 3), np.int8)
    score = logp[0].astype(np.float64).copy()
    lp = logp.astype(np.float64)
    t00, t01, t02 = trans[0]
    t10, t11, t12 = trans[1]
    t20, t21, t22 = trans[2]
    s0, s1, s2 = score
    for t in range(1, T):
        a0, a1, a2 = s0 - t00, s1 - t10, s2 - t20
        if a0 >= a1 and a0 >= a2:
            n0, b0 = a0, 0
        elif a1 >= a2:
            n0, b0 = a1, 1
        else:
            n0, b0 = a2, 2
        a0, a1, a2 = s0 - t01, s1 - t11, s2 - t21
        if a0 >= a1 and a0 >= a2:
            n1, b1 = a0, 0
        elif a1 >= a2:
            n1, b1 = a1, 1
        else:
            n1, b1 = a2, 2
        a0, a1, a2 = s0 - t02, s1 - t12, s2 - t22
        if a0 >= a1 and a0 >= a2:
            n2, b2 = a0, 0
        elif a1 >= a2:
            n2, b2 = a1, 1
        else:
            n2, b2 = a2, 2
        r = lp[t]
        s0, s1, s2 = n0 + r[0], n1 + r[1], n2 + r[2]
        back[t] = (b0, b1, b2)
    out = np.zeros(T, np.int8)
    out[-1] = int(np.argmax([s0, s1, s2]))
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def decode(prob, params, blocks):
    """prob: (T, 3) H/A/D probabilities; blocks: list of (start, stop) decoded independently.
    Frames outside every block are left as 3 (X)."""
    eps = 1e-4
    logp = np.log(np.clip(prob, eps, 1)) * params.get("temp", 1.0)
    logp[:, 2] += params.get("d_bias", 0.0)
    out = np.full(len(prob), 3, np.int8)
    for a, b in blocks:
        out[a:b] = viterbi(logp[a:b], params["c_switch"], params["c_to_dead"], params["c_from_dead"])
    return out

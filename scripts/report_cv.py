"""Per-game CV table (external games): ours vs the organisers' baseline idea (nearest player +
majority vote over +/-50 frames, 'no votes' -> D), on the same degraded held-out games."""
import glob, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import decode, evaluate, dfl_data, degrade as dg
from possession.training import EVAL_SEED

params = json.load(open("models/decoder.json"))["params"]
prof = json.load(open("models/noise_profile.json"))
rows = []
for p in sorted(glob.glob("cache/oof/*.npz")):
    mid = os.path.basename(p)[:-4]
    d = dict(np.load(p)); runs = evaluate.runs_of(d["half"], d["labels"])
    ours = evaluate.score(d["labels"], decode.decode(d["prob"], params, runs), runs)
    g = dg.degrade(dfl_data.load(mid, "ext/dfl", "cache"), prof, EVAL_SEED)
    dd = np.hypot(g.px - g.bx[:, None], g.py - g.by[:, None])
    dd = np.where(np.isfinite(dd), dd, np.inf)
    j = dd.argmin(1); near = np.where(np.isfinite(dd.min(1)), g.team[j], -1)
    w = 50; votes = np.zeros((len(near), 2))
    for k in (0, 1):
        c = np.r_[0, np.cumsum(near == k)]
        idx = np.arange(len(near)); lo = np.clip(idx - w, 0, len(near)); hi = np.clip(idx + w + 1, 0, len(near))
        votes[:, k] = c[hi] - c[lo]
    base = np.where(votes.sum(1) == 0, 2, votes.argmax(1)).astype(np.int8)
    b = evaluate.score(d["labels"], base, runs)
    rows.append((mid, ours, b))
    print(f"| {mid} | {ours['spell_f1']:.3f} | {ours['macro_f1']:.3f} | {ours['n_pred']} / {ours['n_gt']} | {b['spell_f1']:.3f} | {b['macro_f1']:.3f} |", flush=True)
m = lambda k, i: np.mean([r[i][k] for r in rows])
print(f"| **mean** | **{m('spell_f1',1):.3f}** | **{m('macro_f1',1):.3f}** | | {m('spell_f1',2):.3f} | {m('macro_f1',2):.3f} |")

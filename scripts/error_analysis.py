"""Where do spells get lost on the external CV? (recall by GT spell length, transition timing)"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from possession import decode, evaluate, possession_metrics as pm

params = json.load(open(os.environ.get("DEC", "models/decoder.json")))["params"]
buckets = [0, 50, 125, 250, 500, 10 ** 9]
hit = np.zeros(len(buckets) - 1); tot = np.zeros(len(buckets) - 1)
dts = []
for p in sorted(glob.glob("cache/oof/*.npz")):
    d = dict(np.load(p)); runs = evaluate.runs_of(d["half"], d["labels"])
    pred = decode.decode(d["prob"], params, runs)
    gs = pm.possession_spells(d["labels"], runs); ps = pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(gs, ps)
    matched = pm.matched_gt_flags(ov, len(gs), 0.5)
    for (s, e, _), m in zip(gs, matched):
        b = np.searchsorted(buckets, e - s, side="right") - 1
        tot[b] += 1; hit[b] += m
    # transition timing: for each GT change between H and A, nearest predicted change of same type
    g = d["labels"]; tg = np.flatnonzero((g[1:] != g[:-1]) & (g[1:] < 2) & (g[:-1] < 2)) + 1
    tp = np.flatnonzero((pred[1:] != pred[:-1]) & (pred[1:] < 2) & (pred[:-1] < 2)) + 1
    for t in tg:
        if len(tp):
            j = tp[np.argmin(np.abs(tp - t))]
            if abs(j - t) < 125 and pred[j] == g[t]:
                dts.append(j - t)
    # dead starts
print("recall@IoU0.5 by GT spell length:")
for i in range(len(tot)):
    print("  %4.1f-%4.1fs: %d/%d = %.2f" % (buckets[i] / 25, min(buckets[i + 1], 99999) / 25, hit[i], tot[i], hit[i] / max(tot[i], 1)))
dts = np.array(dts)
print("H<->A transition offset pred-gt (frames): median %d, IQR %d..%d, n=%d" % (np.median(dts), *np.percentile(dts, [25, 75]), len(dts)))

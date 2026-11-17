"""spellF1 / macroF1 with the official scorer, for any match with a half index."""
import numpy as np

from . import possession_metrics as pm


def runs_of(half, labels):
    out = []
    for h in np.unique(half):
        if h == 0:
            continue
        idx = np.flatnonzero((half == h) & (labels != 3))
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def score(gt, pred, runs):
    filled = pm.fill_abstentions(pred.astype(np.int8), runs)
    sp = pm.spell_scores(gt, filled, runs)
    mask = np.zeros(len(gt), bool)
    for a, b in runs:
        mask[a:b] = True
    cm = pm.confusion(gt[mask], pred[mask])
    return {"spell_f1": sp["mean_f1"], "macro_f1": float(pm.macro_f1(pm.scored_confusion(cm))),
            "n_gt": sp["n_gt"], "n_pred": sp["n_pred"]}

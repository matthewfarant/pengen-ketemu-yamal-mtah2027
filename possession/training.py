"""Training rows from the degraded external matches."""
import numpy as np

from . import degrade as dg, features as fe

TRAIN_SEEDS = (11, 12, 13)
EVAL_SEED = 99
STEP = 4  # use every 4th frame for training (neighbouring frames are near duplicates)


def training_rows(m, prof, seeds, step=STEP):
    Xs, ys = [], []
    for s in seeds:
        g = dg.degrade(m, prof, s)
        base = fe.base_features(g)
        for us in (0, 1):
            X, names = fe.features(g, base, us)
            y = fe.labels_for(g, us)
            sel = np.arange((s + us) % step, len(y), step)
            sel = sel[y[sel] < 3]
            Xs.append(X[sel]); ys.append(y[sel])
    return np.concatenate(Xs), np.concatenate(ys), names



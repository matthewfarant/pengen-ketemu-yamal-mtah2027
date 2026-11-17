"""Second stage: re-read the first stage's probabilities in their temporal context.

The first stage sees hand-made context windows; the second sees how its own H/A/D beliefs evolve
over the surrounding seconds (before and after), which is where a possession's shape shows. Trained
on out-of-fold first-stage probabilities of the external matches, never on the provided game."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import _roll

PARAMS = dict(learning_rate=0.08, max_iter=200, max_leaf_nodes=31, min_samples_leaf=400,
              l2_regularization=1.0, early_stopping=False, random_state=11)


def feats(prob, us):
    """prob: (T, 3) H/A/D. Point of view 'us' (0 home, 1 away) -> columns us/them/dead."""
    p = prob[:, [us, 1 - us, 2]].astype(np.float64)
    cols = [p]
    for w in (5, 25, 75, 250, 750):
        cols.append(np.stack([_roll(p[:, k], w) for k in range(3)], 1))
    for w in (25, 75, 250):
        for off in (-w // 2, w // 2):
            cols.append(np.stack([_roll(p[:, k], w, off) for k in range(3)], 1))
    X = np.concatenate(cols, 1)
    margin = p[:, 0] - p[:, 1]
    extra = [ _roll(margin, w) for w in (25, 75, 250)]
    return np.concatenate([X, np.stack(extra, 1)], 1).astype(np.float32)


def train(probs, labels, step=3):
    Xs, ys = [], []
    for p, lab in zip(probs, labels):
        for us in (0, 1):
            X = feats(p, us)
            y = np.full(len(lab), 3, np.int8)
            y[lab == 2] = 2; y[lab == us] = 0; y[lab == 1 - us] = 1
            sel = np.arange(us, len(y), step)
            sel = sel[y[sel] < 3]
            Xs.append(X[sel]); ys.append(y[sel])
    clf = HistGradientBoostingClassifier(**PARAMS)
    clf.fit(np.concatenate(Xs), np.concatenate(ys))
    return clf


def predict(clf, prob):
    p0 = clf.predict_proba(feats(prob, 0))
    p1 = clf.predict_proba(feats(prob, 1))
    return 0.5 * (p0 + p1[:, [1, 0, 2]])

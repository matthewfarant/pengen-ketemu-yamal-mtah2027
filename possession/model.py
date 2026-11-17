"""Frame classifier: gradient-boosted trees on the point-of-view features, both points of view
averaged. scikit-learn's HistGradientBoostingClassifier (no system libraries, deterministic for a
fixed random_state)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features as fe

HGB_PARAMS = dict(learning_rate=0.08, max_iter=300, max_leaf_nodes=63, min_samples_leaf=200,
                  l2_regularization=1.0, max_features=0.7, early_stopping=False, random_state=7)


def train(X, y, **kw):
    clf = HistGradientBoostingClassifier(**dict(HGB_PARAMS, **kw))
    clf.fit(X, y)
    return clf


def predict_match(clf, match):
    """(T, 3) probabilities for H, A, D."""
    base = fe.base_features(match)
    X0, _ = fe.features(match, base, 0)
    p0 = clf.predict_proba(X0)
    del X0
    X1, _ = fe.features(match, base, 1)
    p1 = clf.predict_proba(X1)
    # point of view 0: [home, away, dead]; point of view 1: [away, home, dead]
    return 0.5 * (p0 + p1[:, [1, 0, 2]])

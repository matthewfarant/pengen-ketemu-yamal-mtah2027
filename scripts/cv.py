"""Leave-one-match-out cross-validation on the external DFL matches.

Writes out-of-fold H/A/D probabilities for each held-out match (degraded with an unseen seed), so
the decoder can be tuned on matches the classifier never saw. Nothing here touches the provided
game.
"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import dfl_data, degrade as dg, features as fe, model as md

from possession.training import training_rows, TRAIN_SEEDS, EVAL_SEED


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="ext/dfl")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=os.environ.get("OOF_DIR", "cache/oof"))
    ap.add_argument("--matches", nargs="*", default=dfl_data.MATCHES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof = json.load(open("models/noise_profile.json"))
    rows = {}
    for mid in a.matches:
        t = time.time()
        m = dfl_data.load(mid, a.raw, a.cache)
        rows[mid] = training_rows(m, prof, TRAIN_SEEDS)
        print(mid, rows[mid][0].shape, round(time.time() - t), "s", flush=True)
    names = rows[a.matches[0]][2]
    for mid in a.matches:
        t = time.time()
        X = np.concatenate([rows[k][0] for k in a.matches if k != mid])
        y = np.concatenate([rows[k][1] for k in a.matches if k != mid])
        b = md.train(X, y)
        del X, y
        m = dfl_data.load(mid, a.raw, a.cache)
        g = dg.degrade(m, prof, EVAL_SEED)
        p = md.predict_match(b, g)
        np.savez_compressed(os.path.join(a.out, f"{mid}.npz"), prob=p.astype(np.float32),
                            labels=m.labels, half=m.half)
        acc = (p.argmax(1) == m.labels)[m.labels < 3].mean()
        print("fold", mid, "frame acc %.3f" % acc, round(time.time() - t), "s", flush=True)
        if mid == a.matches[0]:
            imp = []
            print("top features:", [n for _, n in imp], flush=True)

"""Train the frame classifier on all seven external DFL matches -> models/classifier.pkl."""
import argparse, json, os, pickle, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import dfl_data, model as md
from possession.training import training_rows, TRAIN_SEEDS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="ext/dfl")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="models/classifier.pkl")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    dfl_data.download(a.raw)  # verifies sizes, fetches only what is missing
    prof = json.load(open("models/noise_profile.json"))
    Xs, ys = [], []
    for mid in dfl_data.MATCHES:
        X, y, _ = training_rows(dfl_data.load(mid, a.raw, a.cache), prof, TRAIN_SEEDS)
        Xs.append(X); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys
    t = time.time()
    clf = md.train(X, y)
    pickle.dump(clf, open(a.out, "wb"))
    print("trained on", X.shape, "in", round(time.time() - t), "s")

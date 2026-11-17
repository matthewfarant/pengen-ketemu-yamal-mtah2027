"""Train the second stage on the out-of-fold first-stage probabilities of all external matches
(cache/oof from scripts/cv.py) -> models/stage2.pkl."""
import glob, os, pickle, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import stage2

data = [dict(np.load(p)) for p in sorted(glob.glob("cache/oof/*.npz"))]
clf = stage2.train([d["prob"] for d in data], [d["labels"] for d in data])
pickle.dump(clf, open("models/stage2.pkl", "wb"))
print("stage2 trained on", len(data), "matches")

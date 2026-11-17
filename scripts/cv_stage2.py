"""Leave-one-match-out for the second stage, on the first stage's out-of-fold probabilities."""
import glob, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import stage2

paths = sorted(glob.glob("cache/oof/*.npz"))
data = [dict(np.load(p)) for p in paths]
os.makedirs("cache/oof2", exist_ok=True)
for i, p in enumerate(paths):
    tr = [d for j, d in enumerate(data) if j != i]
    clf = stage2.train([d["prob"] for d in tr], [d["labels"] for d in tr])
    q = stage2.predict(clf, data[i]["prob"])
    np.savez_compressed(p.replace("oof", "oof2"), prob=q.astype(np.float32),
                        labels=data[i]["labels"], half=data[i]["half"])
    lab = data[i]["labels"]
    print(os.path.basename(p), "acc stage1 %.3f stage2 %.3f" % (
        (data[i]["prob"].argmax(1) == lab)[lab < 3].mean(), (q.argmax(1) == lab)[lab < 3].mean()), flush=True)

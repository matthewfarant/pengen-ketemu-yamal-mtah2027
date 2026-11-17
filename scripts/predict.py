"""Label every frame of the provided game's tactical video. Usage:
    python scripts/predict.py --data $MTAH_DATA_DIR --out submissions/<team>.csv
"""
import argparse, json, os, pickle, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.simplefilter("ignore")
from possession import io_atd, model as md, decode, kickoff, stage2

N_FRAMES = 199319
# graded halves, from games.half_video_blocks("BAR-ATH") in the challenge repo (metadata, not labels)
BLOCKS = [(11185, 90652), (110808, 190198)]
LETTERS = np.array(list("HADX"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "more-than-a-hack-assets"))
    ap.add_argument("--model", default="models/classifier.pkl")
    ap.add_argument("--stage2", default="models/stage2.pkl")
    ap.add_argument("--decoder", default="models/decoder.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probs", default=None, help="optional: save per-frame probabilities (.npz)")
    a = ap.parse_args()
    tac = os.path.join(a.data, "tactical")
    m = io_atd.load(os.path.join(tac, "BAR-ATH_tactical_atd.xml"),
                    os.path.join(tac, "BAR-ATH_tactical_atd.txt"), N_FRAMES)
    half = np.zeros(N_FRAMES, np.int8)
    for h, (s, e) in enumerate(BLOCKS, 1):
        half[s:e] = h
    m.half = half
    clf = pickle.load(open(a.model, "rb"))
    prob = md.predict_match(clf, m)
    if os.path.exists(a.stage2):
        prob = stage2.predict(pickle.load(open(a.stage2, "rb")), prob)
    # before each half's kickoff the game has not started: dead ball
    kos = kickoff.kickoffs_per_block(m, os.path.join(tac, "BAR-ATH_tactical_smart_tagging.xml"), BLOCKS)
    for (s, e), k in zip(BLOCKS, kos):
        prob[s:k] = [0.0, 0.0, 1.0]
    params = json.load(open(a.decoder))["params"]
    codes = decode.decode(prob, params, BLOCKS)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{c}\n" for i, c in enumerate(LETTERS[codes]))
    if a.probs:
        np.savez_compressed(a.probs, prob=prob.astype(np.float32), codes=codes)
    n = {k: int((codes == i).sum()) for i, k in enumerate("HADX")}
    print("kickoffs", kos, "labels", n)

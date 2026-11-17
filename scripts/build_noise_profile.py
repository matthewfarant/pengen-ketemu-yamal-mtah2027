"""Measure the provided tracking's gaps and fragmentation (no labels, no video) ->
models/noise_profile.json, used to degrade the external training data."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from possession import io_atd, noise_profile

N_FRAMES = 199319
BLOCKS = [(11185, 90652), (110808, 190198)]  # graded halves (challenge metadata)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "more-than-a-hack-assets"))
    ap.add_argument("--out", default="models/noise_profile.json")
    a = ap.parse_args()
    tac = os.path.join(a.data, "tactical")
    m = io_atd.load(os.path.join(tac, "BAR-ATH_tactical_atd.xml"),
                    os.path.join(tac, "BAR-ATH_tactical_atd.txt"), N_FRAMES)
    play = np.zeros(N_FRAMES, bool)
    for s, e in BLOCKS:
        play[s:e] = True
    p = noise_profile.measure(m, play)
    noise_profile.save(p, a.out)
    print("ball missing %.3f, players per frame %.2f" % (p["ball_missing_frac"], p["players_per_frame_mean"]))

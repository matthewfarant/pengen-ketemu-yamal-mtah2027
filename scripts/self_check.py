"""Label-free sanity checks of a submission: possessions per half, durations, shares."""
import sys
import numpy as np, pandas as pd
lab = pd.read_csv(sys.argv[1])["label"].values
for h, (s, e) in enumerate([(11185, 90652), (110808, 190198)], 1):
    x = lab[s:e]
    ch = np.flatnonzero(x[1:] != x[:-1]) + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(x)]; v = x[st]; L = (en - st) / 25
    pos = np.isin(v, ["H", "A"])
    print(f"half {h}: possessions {pos.sum()} (H {np.sum(v=='H')}, A {np.sum(v=='A')}), dead spells {np.sum(v=='D')}, "
          f"possession length median {np.median(L[pos]):.1f}s min {L[pos].min():.2f}s, dead median {np.median(L[v=='D']):.1f}s, "
          f"spells <2s {np.sum(L[pos] < 2)}")

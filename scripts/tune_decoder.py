"""Choose the decoder's switch costs on the external matches' out-of-fold probabilities.

Objective: mean spellF1 over the held-out DFL matches (the challenge's ranking metric), macroF1 as
the tie-break. Coordinate search from a coarse start, deterministic."""
import glob, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from possession import decode, evaluate

OOF = sorted(glob.glob(os.environ.get("OOF_GLOB", "cache/oof/*.npz")))
DATA = [dict(np.load(p)) for p in OOF]


def run_one(d, params):
    runs = evaluate.runs_of(d["half"], d["labels"])
    pred = decode.decode(d["prob"], params, runs)
    return evaluate.score(d["labels"], pred, runs)


def objective(params):
    res = Parallel(n_jobs=8)(delayed(run_one)(d, params) for d in DATA)
    sf = np.mean([r["spell_f1"] for r in res]); mf = np.mean([r["macro_f1"] for r in res])
    return sf, mf, res


GRID = {
    "temp": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
    "c_switch": [5, 10, 20, 30, 40, 80],
    "c_to_dead": [5, 10, 20, 40, 60, 80],
    "c_from_dead": [5, 10, 20, 40, 60, 80],
    "d_bias": [-0.5, 0.0, 0.5, 1.0],
}

if __name__ == "__main__":
    best = {"temp": 2.0, "c_switch": 20, "c_to_dead": 56.0, "c_from_dead": 40, "d_bias": 0.5}
    best_s = objective(best)[:2]
    print("start", best, best_s, flush=True)
    for sweep in range(3):
        improved = False
        for k, vals in GRID.items():
            for v in vals:
                if v == best[k]:
                    continue
                p = dict(best, **{k: v})
                s = objective(p)[:2]
                if s > best_s:
                    best, best_s, improved = p, s, True
                    print("better", best, "spellF1 %.4f macroF1 %.4f" % s, flush=True)
        # refine around the current best by halving/doubling steps
        for k in ("c_switch", "c_to_dead", "c_from_dead"):
            for f in (0.7, 1.4):
                p = dict(best, **{k: round(best[k] * f, 2)})
                s = objective(p)[:2]
                if s > best_s:
                    best, best_s, improved = p, s, True
                    print("better", best, "spellF1 %.4f macroF1 %.4f" % s, flush=True)
        if not improved:
            break
    sf, mf, res = objective(best)
    for pth, r in zip(OOF, res):
        print(os.path.basename(pth), {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print("FINAL", best, "spellF1 %.4f macroF1 %.4f" % (sf, mf))
    json.dump({"params": best, "dfl_cv_spell_f1": sf, "dfl_cv_macro_f1": mf},
              open(os.environ.get("DECODER_OUT", "models/decoder.json"), "w"), indent=1)

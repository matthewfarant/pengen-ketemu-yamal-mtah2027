# MTAH 2027 · Phase 1 — who has the ball?

This repository is our entry for FC Barcelona's *More Than A Hackathon* 2027, Phase 1.

- **Input:** automatic tracking of one match (Barça v Athletic).
- **Output:** one label per tactical-video frame: `H` (home possession), `A` (away possession),
  `D` (dead ball) or `X` (outside the graded halves).

**Approach in short:**
1. Train only on **other games**: the Sportec Open DFL dataset (CC BY 4.0), whose per-frame flags
   map exactly onto H/A/D.
2. Before training, degrade those games to look like the provided automatic tracking: the ball is
   missing 41% of the time, tracks are fragmented, and the ball is smoothed and offset.
3. Classify every frame from both teams' points of view.
4. Turn the frame beliefs into possession spells with a Viterbi decoder tuned on the external games.

The details, validation and limitations are in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Reproduce

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
MTAH_DATA_DIR=/path/to/more-than-a-hack-assets bash run.sh "Pengen Ketemu Yamal"
```

- The output is `submissions/Pengen Ketemu Yamal.csv`.
- On a laptop, it takes about 15 minutes.
- It downloads about 2.6 GB of external data into `ext/`.

No match data is in this repository, and none must be: the competition data is licensed for the
competition only.

## Layout

| path | what |
|---|---|
| `possession/io_atd.py`, `io_dfl.py` | Readers for the provided ATD and the external DFL XML |
| `possession/labels.py` | Brings DFL's possession flags to the challenge's "controlled possession" definition |
| `possession/noise_profile.py`, `degrade.py` | Measures how incomplete the provided tracking is (no labels), and degrades the external games to match |
| `possession/features.py` | Point-of-view features with past and future context |
| `possession/model.py` | Gradient-boosted trees (scikit-learn, deterministic) |
| `possession/kickoff.py` | Kickoff instant from the Smart Tagging plus tracking; dead ball before it |
| `possession/decode.py` | Three-state Viterbi with switch costs |
| `possession/possession_metrics.py` | The official scorer, copied from the challenge repository (MIT) |
| `scripts/cv.py`, `tune_decoder.py` | Leave-one-game-out validation and decoder tuning, on external games only |
| `scripts/train_final.py`, `predict.py` | Final training and labelling |
| `possession/broadcast.py` | Tried and dropped: broadcast↔tactical sync and ball-gap filling |

## Credits

- Sportec Open DFL data: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated
  dataset of spatiotemporal and event data in elite soccer.* Scientific Data.
  doi:10.6084/m9.figshare.28196177. CC BY 4.0.
- `possession/possession_metrics.py`: © the challenge organisers, MIT licence.

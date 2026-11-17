# Pengen Ketemu Yamal — possession from noisy tracking, learned on other games

**Team:** Pengen Ketemu Yamal · **Code:** https://github.com/matthewfarant/pengen-ketemu-yamal-mtah2027

Every graded frame of the tactical video gets a label: H (Barça in possession), A (Athletic in
possession) or D (dead ball). Outside the two graded halves the label is X.

**One sentence:** a classifier trained only on seven external Bundesliga games reads 22 noisy
player tracks and an often-missing ball. Before training, those external games are degraded to
look like our tracking. A Viterbi decoder tuned on the same external games then turns the
classifier's per-frame beliefs into possession spells.

## 1. Rules we designed around

The organisers' clarification (Nov 2026) overrides earlier documentation.

| Rule | What it meant for us |
|---|---|
| Train only on external data. | We learn only from the Sportec Open DFL dataset (seven games, CC BY 4.0). |
| Tune only on external data. | Every threshold, cost and hyperparameter is chosen by cross-validation on those games. |
| The public 5-minute ground truth is for checking only. | We used it to check that frames line up and the file is well-formed. We never used it to choose between versions or to tune. |
| No training on the provided video. | Our pipeline reads no pixels and no audio. |
| No hand labels on the provided game. | Every label comes out of the code. Nothing is set per frame, and no constant is tied to a specific frame of the game. |

The provided tracking is used in two ways only:

- **As input to the trained model.**
- **To measure how incomplete it is.** We count ball gap lengths, where on the pitch the ball tends
  to vanish, and how fragmented the player tracks are. These are label-free statistics of the
  input. Their only use is to make the external training data look equally incomplete.

## 2. Data

### Used

**Sportec Open DFL**
- Source: Bassek et al., *Scientific Data* 2025, figshare 10.6084/m9.figshare.28196177, CC BY 4.0.
- Seven games:
  - two from the Bundesliga (J03WMX, J03WN1);
  - five from the 2. Bundesliga (J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9).
- Tracking is TRACAB at 25 Hz.
- Every ball frame carries two flags:
  - `BallPossession`: 1 = home, 2 = away;
  - `BallStatus`: 0 = dead, 1 = alive.
- These map exactly onto H / A / D. No person on our team labelled anything.
- `run.sh` downloads the files.

**Provided game**
- The tactical ATD, as model input.
- The Smart Tagging "Kick Off (Start)" tags, to find each half's kickoff (§4.4).

### Tried and dropped

**Broadcast tracking as a source for the missing ball.**
- We aligned the broadcast tracking to the tactical tracking by searching for the offset that
  best matched the two ball tracks, window by window:
  - one constant offset per half (+4014 and +26920 frames);
  - median ball agreement 0.55 m.
- Replays were rejected by checking that the broadcast players sit on top of the tactical
  players.
- Result: ball coverage on graded frames rose only from 59.3% to 62.6%. Both trackers lose the
  ball in the same situations, so the extra complexity and the risk around replays were not
  worth it.
- The code is in `possession/broadcast.py`.

**Video and audio cues** (referee whistle, broadcast cut-aways at stoppages, shirt colour).
- These are promising, and we measured them while exploring.
- They cannot be trained or tuned without labels from the provided game, which the rules
  forbid. No external game we have comes with both video and per-frame flags.

## 3. From external labels to the challenge's definition

DFL's `BallPossession` flips on touches: about five team spells per minute, many shorter than a
second. The challenge counts a change only on *controlled* possession:

> a simple touch without control, a deflection that does not intercept, an aerial duel without
> subsequent control do not change possession; loose balls stay with the last team

So before training, we hand a team spell shorter than 1 s back to the team that had the ball
before it, when one of these holds:

- the same team has the ball again right after it (`A [H<1s] A`);
- the ball goes dead right after it (`A [H<1s] D`).

These are uncontrolled touches. The 1 s threshold is a reading of "touch without control" (shorter
than any receive-and-play). We did not fit it to any ground truth.

| | possessions per game | median length |
|---|---|---|
| raw DFL, example game | 511 | 3.0 s |
| after the control rule | 395 | 4.8 s |

## 4. Pipeline

### 4.1 Degrading the external tracking (`possession/degrade.py`)

The provided tracking was measured with labels-free statistics only. The external tracking is
then degraded three times per game, with seeds 11, 12 and 13, to match it:

| | provided ATD | clean DFL | degraded DFL |
|---|---|---|---|
| ball missing (graded / play frames) | 41.1% | ~0% | 41.4% |
| ball tracked while off the pitch (share of visible-ball frames) | 3.7% | 25% | 5.3% |
| players per frame | 20.5 | 22 | 20.5 |
| median player speed | 1.36 m/s | 1.41 m/s | 1.41 m/s |
| ball jitter (median 2nd difference) | 0.004 m | 0.000 m | 0.004 m |

How the degradation works:

**Ball gaps**
- A gap starts with a hazard that depends on the pitch zone.
- Its length is drawn from the 146 gap lengths measured on the provided tracking.
- The rate is rescaled until 41% of frames are missing.
- Off-pitch excursions are cut after 0.5 s.

**Ball track**
- A 5-frame moving average.
- A slowly varying offset of about 1 m.

**Players**
- Tracks are cut into fragments whose lengths are drawn from the 1,144 measured fragments.
- Identities are not used anywhere.

### 4.2 Features (`possession/features.py`)

- **Point of view.** Features are computed from one team's point of view ("us" vs "them"), and
  oriented so that "us" attacks towards +x.
  - The attack direction per half comes from where each goalkeeper stands.
  - The model is trained on both points of view, and both are averaged at prediction time, so it
    has no notion of home or away.
- **Per frame, 162 features:**
  - **Ball:** position, distance to the touchlines, off-pitch flag, speed and acceleration.
  - **Ball vs players:** nearest player of each team and the difference, players within 2 m and
    5 m, goalkeeper distance, and whether the ball moves with its nearest player (carrying it).
  - **Team shape:** centroid, length, width, deepest and highest player, players in each box and
    in their own half.
  - **Team motion:** mean forward velocity, mean speed, number of players running.
- **Context.** The labels are offline, so context looks both ways:
  - centred moving means over 0.2 s, 1 s, 3 s and 10 s (plus 30 s for a few);
  - separate 2 s windows into the past and into the future;
  - for the ball's long gaps: time since and until the ball is seen, and where it was (and who
    was nearest) when it vanished and when it reappeared.

### 4.3 Classifier (`possession/model.py`)

- scikit-learn `HistGradientBoostingClassifier`:
  - 300 iterations, 63 leaves, `random_state=7`, deterministic.
- Trained on every 4th frame of 7 games × 3 degradations × 2 points of view.
- Outputs P(us), P(them), P(dead) per frame.

### 4.4 Kickoffs (`possession/kickoff.py`)

- The external games start at kickoff, so the classifier has never seen players walking out for a
  half.
- The game has not started before its kickoff, so those frames are dead ball.
- How the kickoff frame is found:
  1. The Smart Tagging marks a few-second window around each "Kick Off (Start)".
  2. Inside that window, the tracking pins the first frame where the ball, resting near the centre
     spot, starts moving.

### 4.5 Decoder (`possession/decode.py`)

- A three-state Viterbi over H / A / D. It has:
  - a cost for H↔A switches;
  - a cost for going to D;
  - a cost for leaving D;
  - a temperature and a dead-ball bias.
- This is what turns flickering per-frame beliefs into spells. On the ranking metric, a one-frame
  flicker costs as much as a missed possession.
- The costs are chosen by coordinate search to maximise mean spellF1 on the out-of-fold
  predictions of the external games (`scripts/tune_decoder.py`).

## 5. Validation (external games only)

- **Setup:** leave-one-game-out over the seven DFL games.
  - Train on six games, predict the seventh.
  - The held-out game is degraded with an unseen seed (99).
  - Scoring uses the official scorer (`possession/possession_metrics.py`, copied from the challenge
    repository, MIT).

| held-out game | **ours: spellF1** | **ours: macroF1** | predicted / true possessions | baseline* spellF1 | baseline* macroF1 |
|---|---|---|---|---|---|
| J03WMX | 0.635 | 0.919 | 289 / 395 | 0.287 | 0.657 |
| J03WN1 | 0.647 | 0.883 | 325 / 397 | 0.247 | 0.580 |
| J03WOH | 0.701 | 0.916 | 294 / 343 | 0.239 | 0.570 |
| J03WOY | 0.657 | 0.906 | 295 / 400 | 0.244 | 0.556 |
| J03WPY | 0.665 | 0.908 | 269 / 344 | 0.252 | 0.568 |
| J03WQQ | 0.665 | 0.907 | 310 / 359 | 0.236 | 0.530 |
| J03WR9 | 0.678 | 0.913 | 276 / 319 | 0.232 | 0.603 |
| **mean** | **0.664** | **0.907** | | 0.248 | 0.581 |

\*Baseline = the organisers' starter idea: the team of the nearest player to the ball, then a
majority vote over ±50 frames, with "no votes" read as D. It is computed on the same degraded
held-out games.

**A check on the degradation.**
- The baseline scores spellF1 **0.248** on our degraded external games.
- The organisers report **0.243** for the same baseline on the full provided game.
- So the degraded games are about as hard as the real one, which suggests the external
  validation is a fair proxy.

**Results on held-out games:**
- Frame accuracy is 0.907–0.926 per game.
- The decoder's switch costs were chosen on these same out-of-fold predictions, by coordinate
  search over five parameters.

**Where the errors are.** Recall at IoU 0.5, by true spell length:

| true spell length | recall |
|---|---|
| under 2 s | 23% |
| 2–5 s | 50% |
| 5–10 s | 73% |
| over 10 s | 90% |

- Transition timing is unbiased: the median offset is +1 frame, the interquartile range ±15
  frames.
- The remaining loss is short possessions the decoder smooths over.

**Tried and dropped, on the same validation:**

| variant | spellF1 | why dropped |
|---|---|---|
| Second-stage model on the temporal context of the first stage's probabilities | 0.652 | worse |
| Ball-touch and pass-chain features (with ball smoothing matched to the provided track) | 0.666 | Within noise (+0.002). Touch rates differ between the provided and the external tracking (9 vs 21 per minute), which is a transfer risk. |

- **Public window:** score on the published 5 minutes, used only to check alignment and format:
  spellF1 0.629 and macroF1 0.859. For reference, the best baseline on that window scores 0.424 / 0.621. The window carries a ±0.33 bootstrap band and was never used to choose anything.

## 6. Limitations

- **Definition shift.**
  - DFL's flags are not the organisers' ground-truth definition.
  - Their ground truth comes from Metrica's corrected tracking, and appears to confirm control
    rather than mark the first touch.
  - We could only bridge this with the documented definition (§3). Any systematic timing
    difference between the two conventions costs us on spellF1, and the rules rightly leave us no
    way to calibrate it on the game.
- **Domain shift.**
  - The training games are German, the test game is Spanish.
  - Our degradation matches the measured statistics of the provided tracking, but not its
    unknown error modes, such as a ball track locked onto the wrong object.
- **Dead balls with the ball invisible** rely on player movement alone: everyone slows down and
  the shape resets.

## 7. Reproduction

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
MTAH_DATA_DIR=/path/to/more-than-a-hack-assets bash run.sh "Pengen Ketemu Yamal"
# -> "submissions/Pengen Ketemu Yamal.csv" (199,319 rows)
```

- `run.sh` does the following, in order:
  1. measures the noise profile;
  2. downloads the DFL games;
  3. trains the classifier;
  4. labels the game.
- To re-run the tuning of the decoder from scratch: `python scripts/cv.py && python scripts/tune_decoder.py`.
- All randomness is seeded. Running it twice gives byte-identical files (checked with `shasum`).

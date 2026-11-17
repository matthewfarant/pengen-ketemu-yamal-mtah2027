# Literature & prior art — possession / dead-ball inference

This review was compiled on 22 Oct 2026 by two web-research passes.

How sure each claim is:
- **[V]** verified against the primary source in that pass
- **[U]** unverified

It contains no match data, so it is safe to commit.

## 1. Closest analogues to our task

### Biermann, Memmert, Petersen, Raabe (2025), *DMKD* 39:23 — the paper the user flagged
"Contextualization of soccer analysis with tactical periodization and ML."
DOI [10.1007/s10618-025-01092-9](https://doi.org/10.1007/s10618-025-01092-9), CC-BY. [V]

- **Task:** label every frame with a phase: set piece, open-play offence (possession or direct play), open-play defence (pressure or no pressure).
  - Dead frames were removed using the provider's alive flag.
  - Transitions are derived afterwards, not modelled.
- **Possession rule:** "if control is ambiguous, the team that previously had possession keeps it" — the same convention as our ground truth.
- **Features:** 9–17 team-aggregate values per frame:
  - closest-player-to-ball distances
  - distances within and between teams
  - team velocities
  - number of players behind the ball
  - team centroids and x/y variances
- **Model:** FeatGRU (128 units).
- **Results** on 15 matches, first halves only, 5-fold CV, reported as frame accuracy:
  - set piece vs open play: **98%**
  - offence vs defence: **91%**
  - all phases end-to-end: 80%
  - set-piece recall about 60%
- **Not done:** no smoothing (left as future work) and no segment-level metric.
- **Code:** promised in the paper but not found. Data not released.
- **What transfers:** the feature set, the "keep previous possessor" rule, and a small GRU.

### Borghesi et al. (2023), *ESWA* 223:119780 — "Using TCNs to estimate ball possession"
DOI 10.1016/j.eswa.2023.119780. [V]

- **Exactly our three classes** (dead / home / away) from tracking alone.
- Per-object TCN embeddings, aggregated in a way that ignores player order, so **no player IDs needed**.
- **Results:** 86.2% possession accuracy and 89.2% dead/alive accuracy, both per frame.
- **Code:** none found [U].

### Kim et al. (2026), PathCRF, KDD 2026
[arXiv 2602.12080](https://arxiv.org/abs/2602.12080), code [pathcrf](https://github.com/hyunsungkim-ds/pathcrf) (MPL-2.0). [V]

- **Possession state = one edge of the player graph:**
  - self-loop: a player controls the ball
  - u→v: a pass in flight
  - link to an "outside" node: the ball is out
- A CRF with **hard masks on illegal transitions** plus Viterbi decoding.
- **Inputs:** player positions and teams only, no ball, at 5 fps. Trained on Sportec Open DFL.
- **Results:**
  - team-possession frame accuracy **92.4%** with **no ball**
  - masking illegal transitions alone lifted event F1 from 65 to 73–76
  - cross-dataset transfer is weak (66%)

### Kim et al. (2023), Ball Radar, KDD 2023
[arXiv 2306.08206](https://arxiv.org/abs/2306.08206), [code](https://github.com/pientist/ballradar). **No licence**, so reimplement rather than reuse. [V]

- Set Transformer + hierarchical Bi-LSTM: predicts the possessor first, then the ball position.

| Share of ball frames observed | Team possession accuracy | Ball position error |
|---|---|---|
| 0% | 85.9% | 3.7 m |
| 5% | 91.5% | — |
| 10% | 95.0% | — |
| 20% | 97.4% | 1.3 m |

- Our ball is present on about 59% of frames, but in **long gaps**, so inside a gap we are closer to the "no ball" regime, anchored at both ends.

### Capellera et al. (2024), TranSPORTmer, ACCV
[arXiv 2410.17785](https://arxiv.org/abs/2410.17785). Trained on LaLiga data. No code found [U].

- Per-frame state: pass / possession / **uncontrolled** / out.
- **Results:** 80.8% state accuracy with the ball fully hidden.
- Masked-imputation framing.

### Peral et al. (2025, WACV 2026)
[arXiv 2512.19528](https://arxiv.org/abs/2512.19528).

- Adds **tactical-camera player crops** to the trajectories: possessor accuracy rises from 83.8% to 89.6%.
- Team-type tokens add about 7 points of state accuracy.

## 2. Rule-based possession and events (ball visible)

- **Link & Hoernig (2017), *PLOS ONE*** — "Individual ball possession in soccer". [V]
  - **Possession start:** nearest player within a distance threshold **and** a ball acceleration peak of at least 4 m/s² (a kick). F = .83 for possession start.
  - **Possession end:** the ball, projected 1 s ahead, is beyond reach.
  - **Control:** a Bayesian net decides whether the player was in control, from ball-speed variance, how long the ball stays near the player, and the number of opponents within 0.5–5 m.
  - **Preprocessing:** RTS smoothing, remove jumps, drop fragments shorter than 1 s.
  - Their constructs run from individual ball action up to team ball possession.
- **Vidal-Codina et al. (2022), *Sports Engineering*** ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)). [V]
  - **Possession zone:** ball within 0.5–1 m of a player.
  - **Duel zone:** two or more opponents within 1 m of the ball. **A duel overrides possession**, which is our "aerial duels don't count".
  - A minimum number of frames is needed to gain possession.
  - Set-piece triggers from IFAB geometry (kickoff, corner, goal kick, throw-in).
  - Detects over 90% of most event types.
  - **Open implementation:** [databallpy](https://github.com/Alek050/databallpy), `features/player_possession.py` (MIT).
- **ETSY** (KU Leuven, Apache-2.0) and **ELASTIC** (MPL-2.0): event↔tracking synchronisation. Only useful for building labels on public data.

## 3. Dead ball / ball in play

- **Lang, Wild, Isenko, Link (2022), *Sci. Rep.* 12:16291** — in play vs interrupted from **player positions only**. [V]
  - **Features:** positions, speeds and accelerations, plus context windows of ±0.4 to 80 s.
  - **Model:** AdaBoost, with a **3.64 s median filter** afterwards.
  - **Results** on 102 Bundesliga matches: frame accuracy 0.92, F1 0.93.
  - Stoppages longer than 2 s found with 81% precision. Start and end within 2 s for about 80% of stoppages found.
  - **Fails on 2–4 s stoppages and merges short live periods.** That is our risk too.
- **Gudmundsson & Wolle (2014):** ball-out detection F1 0.94 from ball and players [U].
- **SoccerNet Ball Action Spotting 2024/2025** ([2409.10587](https://arxiv.org/abs/2409.10587), [2508.19182](https://arxiv.org/abs/2508.19182)).
  - 12 classes, including Out, Throw-in and Free kick. The 2025 version adds team attribution.
  - Trained on broadcast footage. Video needs an NDA.
  - The winning entry, T-DEED, has [code](https://github.com/arturxe2/T-DEED).
- **Broadcast whistle detection:** narrowband energy at 3.5–4.5 kHz plus spectral-shape checks. About a 19% miss rate in the literature. It marks the start of stoppages, not restarts.
- **Replay detection** in broadcast: a replay almost always follows a stoppage.

## 4. Possession without the ball (team shape)

- **Shaw & Glickman (2019), Barça Sports Analytics Summit**: in-possession formations are wider and longer, with full-backs high. Out-of-possession formations are compact.
- **FIFA EFI phase framework:** a phase has to persist for N frames before it counts — industry precedent for minimum durations.
- Useful features:
  - each team's centroid height
  - length and width
  - number of players goal-side of the ball
  - velocity relative to the attacking direction
  - pressure on the likely carrier
- Reported accuracy for possession without the ball: 86–92% (Kim 2023, Biermann, PathCRF).

## 5. Temporal decoding — the lever for spellF1

| Method | Behaviour | Notes |
|---|---|---|
| HMM + Viterbi with switch cost λ | Equivalent to a Potts / L0 fit. A large λ kills flicker **and** short spells | Easy baseline. Tune λ **on spellF1** |
| **HSMM / explicit-duration Viterbi** | A duration prior per state: minimum about 0.5–0.8 s for possessions, long for dead balls | Cleanest fit for our metric. hsmmlearn, or our own segmental Viterbi O(T·L·K²) at about 5 fps |
| Constrained CRF (PathCRF) | Forbid illegal transitions | e.g. no H→A switch without credible control by A |
| Boundary branch + pooling (ASRF, WACV 2021) | Detect change points, then pool class scores between them | Keeps confident short spells |
| MS-TCN smoothing loss | Truncated-MSE term against over-segmentation | Only if we train a sequence model |
| Change-point detection (ruptures, PELT) | Penalty β ≈ λ; can enforce a minimum segment length | Useful as a second opinion on boundaries |

Spell-F1 is the F1@k metric from action segmentation (Lea et al. 2017). **Select models and parameters by spellF1, never by frame accuracy.**

## 6. Local CV tooling (no cloud)

- **[roboflow/sports](https://github.com/roboflow/sports)**: code is MIT, but **Ultralytics YOLOv8 is AGPL-3.0**.
  - YOLOv8 detectors for player, goalkeeper, referee and ball.
  - Team assignment via SigLIP, UMAP and KMeans.
  - Supports MPS. Do not use hosted Roboflow inference (that would break the no-cloud rule).
- **[WASB-SBDT](https://github.com/nttcom/WASB-SBDT)** (BMVC 2023): a TrackNet-style multi-frame heatmap detector for small balls, with a soccer benchmark. Weights and licence [U].
- **SAHI** tiling: a large recall gain on tiny objects. Apply it only around the predicted ball area or inside gaps.
- **Team colour:** with our kits (navy/garnet vs lilac), a Lab/HSV torso histogram plus a GMM is probably enough. It is cheap and deterministic. Add a reject class for the referee and the green goalkeeper.
- **Local VLMs** (Qwen2.5-VL-7B Apache-2.0; the 3B uses the Qwen licence; Florence-2 MIT):
  - weak on small balls
  - several seconds per frame on an M1
  - at most an offline helper at about 1 fps for questions like "is play stopped?"
- **Determinism:** MPS output is not bit-exact. Cache detector outputs as versioned features, and keep decoding in CPU numpy.
- **M1 Pro throughput** (all [U]):
  - YOLOv8n/s at 1280 px: about 15–40 fps, so 1.5–4 h for the whole match
  - SAHI makes this 5–8× slower

## 7. Public datasets from OTHER matches (allowed for training)

| Dataset | Licence | Labels | Fit |
|---|---|---|---|
| **Sportec Open DFL / IDSSE**: Bassek et al., *Sci. Data* 2025, [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/), figshare 10.6084/m9.figshare.28196177 | CC BY 4.0 | 7 Bundesliga matches, TRACAB 25 Hz. **Per frame `BallPossession` (1 = home, 2 = away) and `BallStatus` (alive/dead)**, plus events. Loaders: floodlight, databallpy, kloppy | **Exactly our label space.** Degrade to our noise: strip IDs, hide the ball in bursts, drop z, add jitter |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | No formal licence [check] | 3 games, 25 Hz, events (BALL OUT, SET PIECE, RECOVERY…) | Possession derivable. Same vendor as our tags |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | MIT | 10 A-League matches, broadcast tracking at 10 fps with `is_detected`, **per-frame possession**, phases of play | Noise profile close to ours |
| [StatsBomb open data](https://github.com/statsbomb/open-data) + 360 | Own licence | Event-level possession | Priors on durations and transitions only |
| PFF FC WC2022 | Form-gated, terms [U] | 64 matches, tracking + events | Check the terms first |
| SoccerNet | NDA for video | Action spots, game-state reconstruction (GSR) | Only if we train video models |

## 8. Broadcast ↔ tactical alignment

1. **Clock OCR on both overlays**, then a per-half linear fit with RANSAC (the SoccerNet approach: [1804.04527](https://arxiv.org/abs/1804.04527)).
   - The broadcast clock is hidden during replays.
   - Our measured result: ±1.5 s.
2. **Refine to the exact frame** by cross-correlating camera-pan velocity (from the homography) or ball/centroid x, with replays masked out.
3. **Audio fingerprinting is possible but weak.** The tactical audio is very quiet ambient sound (normalise it first), while the broadcast mix is dominated by commentary. Whistles and crowd peaks are the shared anchors.

## 9. Traps (from both passes)

- One global smoothing window or λ either keeps flicker or kills the 0.8–4 s spells. Use per-state duration priors.
- **Provider alive flags lag the real whistle** (Vidal-Codina). Allow some slack at boundaries.
- Missing ball frames are **not** random; they cluster in aerial play, crowded boxes and the far touchline. Simulate gaps in bursts.
- Referee vs green goalkeeper breaks 2-cluster colour KMeans.
- Licences: YOLO (AGPL), Ball Radar (none), PFF and StatsBomb (own terms). The organisers must be able to access everything we use.

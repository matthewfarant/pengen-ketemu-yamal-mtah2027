#!/usr/bin/env bash
# One command, from the provided data to the submission file.
#   MTAH_DATA_DIR=/path/to/more-than-a-hack-assets bash run.sh [team_name]
# Downloads the external training data (Sportec Open DFL, CC BY 4.0, ~2.6 GB), measures how
# incomplete the provided tracking is, trains on the external games only, and labels the game.
set -euo pipefail
cd "$(dirname "$0")"
TEAM="${1:-team}"
DATA="${MTAH_DATA_DIR:-more-than-a-hack-assets}"
python scripts/build_noise_profile.py --data "$DATA"
python scripts/train_final.py
python scripts/predict.py --data "$DATA" --out "submissions/${TEAM}.csv"

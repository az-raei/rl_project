#!/usr/bin/env bash
# One command for a whole experiment on Kaggle: tests -> train -> adapt (per task) -> collect results.
#
#   bash scripts/kaggle_run.sh <name> [overrides...]
#   bash scripts/kaggle_run.sh baseline-k16 model.K=16 train.steps=5000
#   TASKS="reach-v3" SKIP_TESTS=1 bash scripts/kaggle_run.sh quick model.K=8 train.steps=2000
#
# The same overrides are passed to `train` and `adapt` so they agree on the checkpoint name.
# Results (logs + tables) are copied to runs/<name>/ ; push them with scripts/push_results.sh <name>.
set -euo pipefail
NAME="${1:?usage: kaggle_run.sh <experiment-name> [overrides...]}"; shift || true
TASKS="${TASKS:-reach-v3 pick-place-v3}"

if [ "${SKIP_TESTS:-0}" != "1" ]; then python tests/run_tests.py; fi
for TASK in $TASKS; do
  python main.py train "tasks=[$TASK]" "$@"
  python main.py adapt "tasks=[$TASK]" "$@"
done

mkdir -p "runs/$NAME"
cp -r results/logs results/tables "runs/$NAME/"
{ echo "experiment: $NAME"; echo "overrides: $*"; echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"; } > "runs/$NAME/INFO.txt"
echo "== done. results in runs/$NAME/ =="

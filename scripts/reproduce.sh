#!/usr/bin/env bash
# Reproduce all experiments (train -> adapt -> active querying -> IQL -> report) and collect results in runs/<name>/.
#
#   bash scripts/reproduce.sh <name> [quick|full] [stage ...]
#   bash scripts/reproduce.sh try1 quick                    # everything, small budgets (smoke / first look)
#   bash scripts/reproduce.sh try1 quick tests baselines    # only some stages
#   TASKS=reach-v3 bash scripts/reproduce.sh try1 quick
#
# stages (default = all, in this order):
#   tests       unit tests incl. the torch tests and an end-to-end fake-data pipeline test
#   baselines   ours | PrefMMT encoder | PrefMMT pooled (single reward) | stats-MLP          (train + adapt)
#   ablations   K in {4,6,8,16,32}, temporal module, prior strength l2
#   querying    random vs ensemble-disagreement vs entropy, trains 4 extra ensemble seeds
#   iql         relabel -> IQL -> ground-truth evaluation (needs `pip install metaworld`; needs `baselines` first)
#   report      tables (markdown) + figures in results/
# quick = up to 2000 training steps, small IQL;  full = up to 6000 training steps (early stopping), IQL 100k steps.
# Run ONE stage at a time if a stage fails, and send me the last ~40 lines of output.
set -euo pipefail
NAME="${1:?usage: reproduce.sh <name> [quick|full] [stage ...]}"; shift
PROFILE="${1:-quick}"; shift || true
STAGES="${*:-tests baselines ablations querying iql report}"
TASKS="${TASKS:-reach-v3,pick-place-v3}"
case "$PROFILE" in
  quick) STEPS=2000;  IQL=(iql.total_steps=20000 iql.eval_episodes=5 iql.users_per_group=1 "iql.budgets=[100]")
         ABL=("exp.k_values=[4,8,16]" "exp.temporal_values=[gru]" "exp.l2_values=[0.1,10]");;
  full)  STEPS=6000;  IQL=(iql.total_steps=100000 iql.eval_episodes=10 iql.users_per_group=2)
         ABL=();;
  *) echo "profile must be quick or full"; exit 1;;
esac
T="tasks=[$TASKS]"

for st in $STAGES; do
  echo; echo "=================== stage: $st ($PROFILE) ==================="
  case "$st" in
    tests)     python tests/run_tests.py ;;
    baselines) python main.py baselines "$T" "exp.steps=$STEPS" ;;
    ablations) python main.py ablations "$T" "exp.steps=$STEPS" "${ABL[@]}" ;;
    querying)  python main.py querying  "$T" "exp.steps=$STEPS" ;;
    iql)       python -c "import metaworld" 2>/dev/null || pip install -q metaworld
               python main.py iql "$T" "${IQL[@]}" ;;
    report)    python main.py report "$T" ;;
    *) echo "unknown stage: $st"; exit 1;;
  esac
done

mkdir -p "runs/$NAME"
cp -r results/logs results/tables "runs/$NAME/" 2>/dev/null || true
[ -d results/figures ] && cp -r results/figures "runs/$NAME/" || true
{ echo "experiment: $NAME ($PROFILE)"; echo "stages: $STAGES"; echo "tasks: $TASKS"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"; } > "runs/$NAME/INFO.txt"
echo; echo "== done. results in runs/$NAME/  (push with: bash scripts/push_results.sh $NAME) =="

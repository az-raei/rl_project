#!/usr/bin/env bash
# Push runs/<name>/ to a NEW branch results-<name> of your GitHub repo (never touches main).
# Needs a fine-grained GitHub token (Contents: read+write on this one repo) in $GH_TOKEN.
#   bash scripts/push_results.sh baseline-k16
set -euo pipefail
NAME="${1:?usage: push_results.sh <experiment-name>}"
: "${GH_TOKEN:?set GH_TOKEN first (Kaggle: Add-ons > Secrets)}"
SLUG="$(git config --get remote.origin.url | sed -E 's#^(https://([^@/]*@)?|git@)github.com[:/]##; s#\.git$##')"
git config user.email "kaggle-run@users.noreply.github.com"
git config user.name  "kaggle-run"
git checkout -q -B "results-$NAME"
git add -f "runs/$NAME"
git commit -q -m "results: $NAME" || echo "(nothing new to commit)"
git push -q --force "https://x-access-token:${GH_TOKEN}@github.com/${SLUG}.git" "results-$NAME" 2>&1 | sed "s/${GH_TOKEN}/***/g"
echo
echo "pushed. Paste this link to Claude:"
echo "https://github.com/${SLUG}/tree/results-${NAME}/runs/${NAME}/tables"

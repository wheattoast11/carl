#!/bin/bash
# Commit this directory onto carl branch experiments/c3-view-breakout-20261010 through a temporary index; the shared working tree is never touched.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
REPO=/var/home/zero/carl
BRANCH=experiments/c3-view-breakout-20261010
PREFIX=experiments/encoder-regime-20261010/c3-view-breakout
export GIT_INDEX_FILE="$(mktemp)"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
cd "$REPO"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then PARENT=$(git rev-parse "$BRANCH"); else PARENT=$(git rev-parse fcb8368); fi
git read-tree "$PARENT"
for f in c3_lib.py c3_ceiling.py c3_train.py c3_transductive.py c3_table.py c3_budget_judge.py c3_rpm_corpus.py pca_large.py commit.sh batch*.sh \
         ceiling.json transductive.json pca-scaling.json pca-man1.json pca-large.json pca-rpm.json budget-judge.json risk-checks.json full-budget-judge.json population-k-sweep.json val-anchor-judge.json c3_val_anchor_judge.py c3-table.json man1-mask.npy \
         predictions.jsonl rank-arm-byte-budget.patch receipt-*.json views-*.f32 head-*.pt view-head-256.f32 view-head-256.json view-head-512.f32 view-head-512.json val-anchor-judge-512.json man1-scaling.json work-admission-handoff.md fit-*.log transductive.log corpus-rpm.log \
         corpus-rpm/receipt.json corpus-rpm/train-corpus.jsonl strix-scratch/strix-mind/lib/rank_arm.py strix-scratch/strix-mind/tests/test_rank_arm.py; do
  for p in "$SRC"/$f; do
    [ -f "$p" ] || continue
    blob=$(git hash-object -w "$p")
    git update-index --add --cacheinfo 100644 "$blob" "$PREFIX/${p#"$SRC"/}"
  done
done
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "${1:-experiments: C3 128-view breakout study}")
git update-ref "refs/heads/$BRANCH" "$COMMIT"
echo "$COMMIT"

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
for f in c3_lib.py c3_ceiling.py c3_train.py c3_transductive.py c3_table.py commit.sh batch1.sh batch2.sh \
         ceiling.json transductive.json pca-scaling.json pca-man1.json c3-table.json man1-mask.npy \
         receipt-*.json views-*.f32 fit-*.log transductive.log; do
  for p in "$SRC"/$f; do
    [ -f "$p" ] || continue
    blob=$(git hash-object -w "$p")
    git update-index --add --cacheinfo 100644 "$blob" "$PREFIX/$(basename "$p")"
  done
done
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "${1:-experiments: C3 128-view breakout study}")
git update-ref "refs/heads/$BRANCH" "$COMMIT"
echo "$COMMIT"

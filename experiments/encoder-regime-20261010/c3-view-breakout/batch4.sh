#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag large-rank128-mlp --corpus large --loss rank --hidden 1536 --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 2500 --predicted 0.855 --judge > fit-large-rank128-mlp.log 2>&1
$PY c3_train.py --tag large-rank128-linear --corpus large --loss rank --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 2000 --predicted 0.845 --judge > fit-large-rank128-linear.log 2>&1
echo BATCH4-DONE > batch4.done

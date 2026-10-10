#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f batch7.done ]; do sleep 5; done
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag kd256-linear --width 256 --loss kd --pop 1024 --groups 1 --steps 3000 --predicted 0.905 --judge --element-bytes 1 > fit-kd256-linear.log 2>&1
$PY c3_train.py --tag rank256-linear-lr1e-4 --width 256 --loss rank --hard 100 --temp 0.01 --lr 1e-4 --man1-frac 0.7 --pop 256 --groups 4 --steps 1500 --predicted 0.895 --judge --element-bytes 1 > fit-rank256-linear-lr1e-4.log 2>&1
$PY c3_train.py --tag kd256-mlp --width 256 --loss kd --hidden 1536 --pop 1024 --groups 1 --steps 3000 --predicted 0.91 --judge --element-bytes 1 > fit-kd256-mlp.log 2>&1
echo BATCH8-DONE > batch8.done

#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag kd224-linear --width 224 --loss kd --pop 1024 --groups 1 --steps 3000 --predicted 0.904 --judge --element-bytes 1 > fit-kd224-linear.log 2>&1
$PY c3_train.py --tag kd256-linear-man1 --width 256 --loss kd --man1-frac 0.7 --pop 1024 --groups 1 --steps 3000 --predicted 0.922 --judge --element-bytes 1 > fit-kd256-linear-man1.log 2>&1
$PY c3_train.py --tag kd192-linear --width 192 --loss kd --pop 1024 --groups 1 --steps 3000 --predicted 0.89 --judge --element-bytes 1 > fit-kd192-linear.log 2>&1
$PY c3_train.py --tag kd320-linear --width 320 --loss kd --pop 1024 --groups 1 --steps 3000 --predicted 0.935 --judge --element-bytes 1 > fit-kd320-linear.log 2>&1
echo BATCH9-DONE > batch9.done

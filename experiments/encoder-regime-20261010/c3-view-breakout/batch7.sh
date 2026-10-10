#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag rank256-mlp-man1 --width 256 --loss rank --hidden 1536 --hard 100 --temp 0.01 --man1-frac 0.7 --pop 256 --groups 4 --steps 2000 --predicted 0.905 --judge --element-bytes 1 > fit-rank256-mlp-man1.log 2>&1
$PY c3_train.py --tag rank256-linear-man1 --width 256 --loss rank --hard 100 --temp 0.01 --man1-frac 0.7 --pop 256 --groups 4 --steps 1500 --predicted 0.90 --judge --element-bytes 1 > fit-rank256-linear-man1.log 2>&1
$PY c3_train.py --tag rank320-mlp-man1 --width 320 --loss rank --hidden 1536 --hard 100 --temp 0.01 --man1-frac 0.7 --pop 256 --groups 4 --steps 2000 --predicted 0.928 --judge --element-bytes 1 > fit-rank320-mlp-man1.log 2>&1
$PY c3_train.py --tag rank384-mlp-man1 --width 384 --loss rank --hidden 1536 --hard 100 --temp 0.01 --man1-frac 0.7 --pop 256 --groups 4 --steps 2000 --predicted 0.955 --judge --element-bytes 1 > fit-rank384-mlp-man1.log 2>&1
echo BATCH7-DONE > batch7.done

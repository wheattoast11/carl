#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f batch5.done ]; do sleep 10; done
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag rank128-mlp-man1-0.9 --loss rank --hidden 1536 --hard 100 --temp 0.01 --man1-frac 0.9 --pop 256 --groups 4 --steps 2000 --predicted 0.845 --judge > fit-rank128-mlp-man1-0.9.log 2>&1
$PY c3_train.py --tag rank128-mlp-man1-0.5 --loss rank --hidden 1536 --hard 100 --temp 0.01 --man1-frac 0.5 --pop 256 --groups 4 --steps 2000 --predicted 0.848 --judge > fit-rank128-mlp-man1-0.5.log 2>&1
echo BATCH6-DONE > batch6.done

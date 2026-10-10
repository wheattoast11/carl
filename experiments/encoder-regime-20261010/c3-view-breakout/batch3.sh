#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f batch2.done ]; do sleep 10; done
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag rank128-man1-0.7 --loss rank --hard 100 --temp 0.01 --man1-frac 0.7 --pop 256 --groups 4 --steps 1500 --predicted 0.825 --judge > fit-rank128-man1-0.7.log 2>&1
$PY c3_train.py --tag rank128-mlp-hard100-t01 --loss rank --hidden 1536 --hard 100 --temp 0.01 --pop 256 --groups 4 --steps 2000 --predicted 0.842 --judge > fit-rank128-mlp-hard100-t01.log 2>&1
echo BATCH3-DONE > batch3.done

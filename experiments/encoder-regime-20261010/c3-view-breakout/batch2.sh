#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f batch1.done ]; do sleep 10; done
export OMP_NUM_THREADS=6
PY=/var/home/zero/carl/.venv/bin/python
$PY c3_train.py --tag rank128-n1500 --loss rank --hard 100 --temp 0.01 --subset 1500 --pop 256 --groups 4 --steps 1500 --predicted 0.80 --judge > fit-rank128-n1500.log 2>&1
$PY c3_train.py --tag rank128-n3000 --loss rank --hard 100 --temp 0.01 --subset 3000 --pop 256 --groups 4 --steps 1500 --predicted 0.815 --judge > fit-rank128-n3000.log 2>&1
$PY c3_train.py --tag rank128-hard100-t01-long --loss rank --hard 100 --temp 0.01 --lr 5e-4 --pop 256 --groups 4 --steps 3000 --predicted 0.835 --judge > fit-rank128-hard100-t01-long.log 2>&1
echo BATCH2-DONE > batch2.done

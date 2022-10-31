#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -J ami
#BSUB -o %J.out
#BSUB -e %J.err
python -u train_lm.py >> out

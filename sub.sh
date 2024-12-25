#!/bin/bash
#SBATCH -J shenxy
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH -A F00120230017
##################################################################

module load cuda11.3/toolkit/11.3.0

function Func1(){
    python main_rebuild.py
}

Func1

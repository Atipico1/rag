#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --time 20:00:00

cd /home/seongilpark/rag
conda init bash
conda activate rag
python wiki.py
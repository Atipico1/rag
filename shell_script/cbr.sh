#!/bin/sh
#SBATCH -J cbr-faiss
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:0
#SBATCH --mem=100G
#SBATCH -o /home/seongilpark/rag/log/cbr-faiss.out
#SBATCH -e /home/seongilpark/rag/log/cbr-faiss.err
#SBATCH --time 12:00:00

python data.py \
    --qa_dataset Seongill/NQ_missing_5_masked \
    --qa_split train \
    --num_cbr 5 \
    --batch_size 512 \
    --output NQ_random
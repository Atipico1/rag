#!/bin/sh
#SBATCH -J nq_mrc
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH -o /home/seongilpark/rag/log/train_nq_mrc_cbr.out
#SBATCH -e /home/seongilpark/rag/log/train_nq_mrc_cbr.err
#SBATCH --time 60:00:00

python train.py \
    --dataset_name Seongill/nq_cbr\
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_contexts 1 \
    --cal_max_len True \
    --save_steps 500 \
    --run_name nq_mrc_cbr \
    --output_dir /data/seongil/rag/nq_mrc_cbr_checkpoints \
    --push_to_hub true
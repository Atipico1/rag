#!/bin/sh
#SBATCH -J 3
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -o /home/seongilpark/rag/log/eval-3-cbr.out
#SBATCH -e /home/seongilpark/rag/log/eval-3-cbr.err
#SBATCH --time 10:00:00

CUDA_VISIBLE_DEVICES=3 python eval_adv.py \
    --model microsoft/Orca-2-13b \
    --task cbr \
    --dataset_size 400 \
    --subset in_topic \
    --num_cases 7 \
    --test False \
    --batch_size 1

# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --model microsoft/Orca-2-13b \
#     --task oracle \
#     --dataset_size 400 \
#     --subset random \
#     --num_cases 3 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --model microsoft/Orca-2-13b \
#     --task baseline \
#     --dataset_size 400 \
#     --subset in_topic \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --model microsoft/Orca-2-13b \
#     --task baseline \
#     --dataset_size 400 \
#     --subset random \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task oracle \
#     --dataset_size 400 \
#     --subset random \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task oracle \
#     --dataset_size 400 \
#     --subset in_topic \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task baseline \
#     --dataset_size 400 \
#     --subset random \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task baseline \
#     --dataset_size 400 \
#     --subset in_topic \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task s2a \
#     --dataset_size 400 \
#     --subset random \
#     --num_cases 5 \
#     --test False
# CUDA_VISIBLE_DEVICES=2,3 python eval_adv.py \
#     --task s2a \
#     --dataset_size 400 \
#     --subset in_topic \
#     --num_cases 5 \
#     --test False
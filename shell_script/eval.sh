#!/bin/sh
#SBATCH -J gpt
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:0
#SBATCH --mem=50G
#SBATCH -o /home/seongilpark/rag/log/eval.out
#SBATCH -e /home/seongilpark/rag/log/eval.err
#SBATCH --time 60:00:00

python eval.py \
    --dataset_name Seongill/NQ_missing_5 \
    --num_contexts 5 \
    --instruction find \
    --q_to_cbrs NQ_random \
    --cbr True \
    --num_cbr_cases 5 \
    --cbr_perturb False \
    --num_perturb_cases 0 \
    --cbr_perturb_type missing \
    --test False \
    --test_size 10

# custom_questions file path
# filter_option [include, exclude] Seongill/NQ_missing_5 9
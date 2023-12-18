#!/bin/sh
#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o /home/seongilpark/rag/log/baseline_w4-e3.out
#SBATCH -e /home/seongilpark/rag/log/baseline_w4-e3.err
#SBATCH --time 12:00:00

cd ~/rag

# CUDA_VISIBLE_DEVICES=3 python run_exp.py \
#  --num_wiki 4 \
#  --size 3610 \
#  --bs 4 \
#  --num_examples 3 \
#  --prompt find \
#  --model_name baseline \
#  --ex_type cbr_exact \
#  --unanswerable true \
#  --lm mistral-instruct \
#  --test false \
#  --data_file TotalPromptSet-all_ex_20.joblib\
#  --max_tokens 100
CUDA_VISIBLE_DEVICES=3 python run_exp.py \
 --num_wiki 6 \
 --size 3610 \
 --bs 8 \
 --num_examples 0 \
 --prompt find \
 --model_name baseline \
 --ex_type cbr_exact \
 --unanswerable true \
 --lm mistral-instruct \
 --test false \
 --data_file TotalPromptSet-all_ex_20.joblib\
 --max_tokens 100
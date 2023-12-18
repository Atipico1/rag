#!/bin/sh
#SBATCH -J hybrid-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o /home/seongilpark/rag/log/2_hybrid_adversarial.out
#SBATCH -e /home/seongilpark/rag/log/2_hybrid_adversarial.err
#SBATCH --time 13:00:00

cd ~/rag
CUDA_VISIBLE_DEVICES=3 python run_exp.py \
 --num_wiki 5 \
 --size 3610 \
 --bs 4 \
 --num_examples 1 \
 --selective_perturbation swap_context \
 --model_name hybrid \
 --prompt find \
 --ex_type cbr_exact \
 --lm mistral-instruct \
 --max_tokens 100
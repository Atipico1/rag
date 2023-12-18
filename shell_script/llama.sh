#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o /home/seongilpark/rag/log/RALM_ctx1.out
#SBATCH -e /home/seongilpark/rag/log/RALM_ctx1.err
#SBATCH --time 09:00:00

cd ~/rag
CUDA_VISIBLE_DEVICES=2 python run_exp.py \
 --num_wiki 2 \
 --size 3610 \
 --bs 8 \
 --num_examples 0 \
 --prompt find \
 --ex_type cbr_exact \
 --unanswerable false \
 --lm Llama-2-7b-hf \
 --model_name baseline \
 --max_tokens 10 \
 --prefix ralm-reproduce

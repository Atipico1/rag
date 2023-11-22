#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH -o log/exp.out
#SBATCH -e log/exp.err
#SBATCH --time 04:00:00

cd /home/seongilpark/rag
conda init bash
conda activate data
python run_exp.py \
 --num_wiki 10 \
 --size 1735 \
 --bs 20 \
 --cache true \
 --num_examples 15 \
 --prompt find \
 --ex_type cbr_exact \
 --unanswerable true \
 --mode single \
 --model_name baseline \
 --test false \
 --skip_gpt false \
 --prompt_test true \
 --skip_wandb false \
 --data_path datasets/TotalPromptSet-all_ex_20_incorrect.joblib \
 --max_tokens 100

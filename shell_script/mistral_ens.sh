#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o /home/seongilpark/rag/log/mistral-3-ensemble-baseline.out
#SBATCH -e /home/seongilpark/rag/log/mistral-3-ensemble-baseline.err
#SBATCH --time 12:00:00

cd ~/rag
python run_exp.py \
 --num_wiki 3 \
 --size 3610 \
 --bs 4 \
 --num_examples 0 \
 --prompt base \
 --ex_type cbr_exact \
 --unanswerable true \
 --lm mistral-instruct \
 --model_name baseline \
 --test false \
 --data_file TotalPromptSet-all_ex_20.joblib\
 --max_tokens 10 \
 --ensemble true \
 --formatting true \
 --prefix ensemble-base
 
 ## Mistral-2511_ex_20_incorrect.joblib
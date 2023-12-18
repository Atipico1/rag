#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:0
#SBATCH --mem=40G
#SBATCH -o /home/seongilpark/rag/log/gpt2.out
#SBATCH -e /home/seongilpark/rag/log/gpt2.err
#SBATCH --time 09:00:00

cd ~/rag
python run_exp.py \
 --num_wiki 10 \
 --size 1735 \
 --bs 20 \
 --num_examples 0 \
 --prompt find \
 --ex_type cbr_exact \
 --unanswerable true \
 --lm gpt-3.5-turbo-instruct \
 --model_name ours \
 --test false \
 --data_file TotalPromptSet-all_ex_20_incorrect.joblib \
 --max_tokens 100 \
 --skip_model_output true \
 --perturb_testset true \
 --perturb_testset_op confilct 

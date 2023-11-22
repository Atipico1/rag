#!/bin/sh
#SBATCH -J seongil-fewshot
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=60G
#SBATCH -o /home/seongilpark/rag/log/data-fortest2.out
#SBATCH -e /home/seongilpark/rag/log/data-fortest2.err
#SBATCH --time 12:00:00

cd /home/seongilpark/rag
conda init bash
conda activate data
python run_data.py \
 --squad_cache true \
 --nq_cache true \
 --num_ex 20 \
 --num_fewshot 5 \
 --num_context 20 \
 --bs 64 \
 --device cuda \
 --filter_wh true \
 --filter_same_answer true \
 --filter_same_question true \
 --custom_name all_ex_20 \
 --masking_model spacy \
 --ent_swap true \
 --swap_context_method similar \
 --test true
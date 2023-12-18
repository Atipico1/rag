#!/bin/sh
#SBATCH -J contriever
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH -o /home/seongilpark/rag/log/contriever.out
#SBATCH -e /home/seongilpark/rag/log/contriever.err
#SBATCH --time 20:00:00

cd /data/seongil/contriever
python passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data nq/train.json \
    --per_gpu_batch_size 128 \
    --output_dir /data/seongil/contriever/contriever_nq
python passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data /data/seongil/contriever/nq/test.json \
    --per_gpu_batch_size 128 \
    --output_dir /data/seongil/contriever/contriever_nq
python passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages psgs_w100.tsv \
    --per_gpu_batch_size 128 \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data /data/seongil/contriever/nq/dev.json \
    --output_dir /data/seongil/contriever/contriever_nq
import argparse
from typing import List
from dataset import masking
from utils import str2bool, update_context_with_substitution_string
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, DPRQuestionEncoder
from tqdm import tqdm
import numpy as np
import spacy 
from copy import deepcopy
import torch
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import List, Dict
from index import build_multiple_indexes
import json
import joblib
import random

WH_WORDS = ["what", "when", "where", "who", "why", "how","which","whom"]
TYPE_DATA_MAP ={
    "missing":"Seongill/squad_missing_answer",
    "conflict":"Seongill/squad_conflict_v2_under_150_with_substitution"
}
TYPE_SPILT_MAP = {
    "missing":"train",
    "conflict":"train"
}

def match_case(qa_dataset: Dataset, multiple_index, args):
    cnt = 0
    output = []
    for row in tqdm(qa_dataset, desc="CASE Matching..."):
        head_word = row["question"].strip().lower().split()[0]
        if (head_word not in WH_WORDS) or (head_word not in multiple_index.keys()):
            head_word = "original"
        index, id2q = multiple_index[head_word]["index"], multiple_index[head_word]["id2q"]
        query = np.array([row["query_embedding"]]).astype("float32")
        distances, indices = index.search(query, args.num_cbr)
        cases = []
        for dist, idx in zip(distances[0], indices[0]):
            matched_row = id2q[idx]
            matched_row.update({"distance":str(dist)})
            cases.append(matched_row)
            cnt += 1
            if args.printing:
                if cnt % (len(qa_dataset) // 5) == 0:
                    print("Original Question: ", row["question"])
                    for k, v in matched_row.items():
                        print(f"Matched {k}:{v}")
                    print("-"*100)
        output.append(cases)
    return output

def query_masking(nlp, dataset: Dataset, args):
    ctxs = dataset["question"]
    result = []
    for i in tqdm(range(0, len(ctxs), args.batch_size), desc="Masking..."):
        batch = ctxs[i:i+args.batch_size]
        batch_docs = list(nlp.pipe(batch, batch_size=args.batch_size))
        masked_quries = [masking(doc, "spacy") for doc in batch_docs]
        result.extend(masked_quries)
    assert len(result) == len(ctxs), "Length doesn't match"
    return dataset.add_column("masked_query", result)

def query_embedding(model, tokenizer, dataset: Dataset, args):
    queries = dataset["masked_query"]
    result = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Embedding..."):
        batch = queries[i:i+args.batch_size]
        output = tokenizer(batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [args.batch_size, hidden_dim]
        result.extend([emb for emb in embeddings])
    assert len(result) == len(queries), "Length doesn't match"
    return dataset.add_column("query_embedding", result)

def remove_duplicate(data: Dataset):
    masked_queries: list[str] = data["masked_query"]
    ids: list[str] = data["id"]
    buffer = dict()
    result_idxs = []
    for uid, query in zip(ids, masked_queries):
        if not buffer.get(query):
            buffer[query] = True
            result_idxs.append(uid)
    def filter_condition(example):
        return example['id'] not in result_idxs
    filtered_data = data.filter(filter_condition)
    return filtered_data

def _preprocess(dataset: Dataset, args):
    if "query_embedding" not in dataset.column_names:
        nlp = spacy.load("en_core_web_trf")
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
        dataset = query_masking(nlp, dataset, args)
        dataset = query_embedding(model, tokenizer, dataset, args)
        if not args.test:
            dataset.push_to_hub(f"{args.qa_dataset}_masked")
        return dataset
    else:
        return dataset

def make_row(row, cbr_type) -> dict:
    if cbr_type == "conflict":
        return {"question":row["question"],
                "original_context":row["context"],
                "context_with_random_answer":row["random_answer"],
                "context_with_similar_answer":row["similar_answer"],
                "context_with_random": update_context_with_substitution_string(row["rewritten_context"], row["answers"]["text"], row["random_answer"]),
                "context_with_similar": update_context_with_substitution_string(row["rewritten_context"], row["answers"]["text"], row["similar_answer"]),
                "new_answer": "conflict",
                "original_answer": row["answers"]["text"]}
    elif cbr_type == "missing":
        return {"question":row["question"],
                "original_context":row["context"],
                "random_context":row["random_answer"],
                "similar_context":row["similar_answer"],
                "hybrid_context":row["similar_answer_v2"],
                "new_answer": "unanswerable",
                "original_answer": row["answers"]["text"]}
    elif cbr_type == "adv":
        #["question", "answer", "answer_sent", "new_answer_sent", "new_answer_chunk", "similar_answer", "answer_chunk", "query_embedding"]
        return {"question":row["question"],
                "answer_chunk":row["answer_chunk"],
                "new_answer_chunk":row["new_answer_chunk"],
                "new_context": row["answer_chunk"]+"\n"+row["new_answer_chunk"],
                "similar_answer":row["similar_answer"],
                "answer":row["answer"]}
    else:
        return {"question":row["question"],
                "context":row["context"],
                "answer":row["answers"]["text"][0]}

def main(args):
    conflict_dataset = load_dataset(TYPE_DATA_MAP["conflict"], split="train")
    missing_dataset = load_dataset(TYPE_DATA_MAP["missing"], split="train")
    adversary_dataset = load_dataset("Seongill/squad_adversarial_thres1", split="train")
    original_dataset = load_dataset("Seongill/SQuAD_unique_questions", split="train")
    qa_dataset = load_dataset(args.qa_dataset, split=args.qa_split)
    if args.test:
        conflict_dataset = conflict_dataset.select(range(10000))
        original_dataset = original_dataset.select(range(10000))
        missing_dataset = missing_dataset.select(range(10000))
        adversary_dataset = adversary_dataset.select(range(10000))
        qa_dataset = qa_dataset.select(range(5))
    qa_dataset = _preprocess(qa_dataset, args)
    print("Dataset Loaded")
    print(f"Conflict: {len(conflict_dataset)}")
    print(f"Original: {len(original_dataset)}")
    print(f"Missing : {len(missing_dataset)}")
    print(f"Adversary : {len(adversary_dataset)}")
    print(f"{args.qa_dataset}: {len(qa_dataset)}")
    sub_conflict, sub_origin, sub_missing, sub_adv = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in conflict_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        sub_conflict[head_word].append((make_row(row, "conflict"), row["query_embedding"]))
    for row in original_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        sub_origin[head_word].append((make_row(row, "original"),row["query_embedding"]))
    for row in missing_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        sub_missing[head_word].append((make_row(row, "missing"), row["query_embedding"]))
    for row in adversary_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        sub_adv[head_word].append((make_row(row, "adv"), row["query_embedding"]))
    
    # multiple_indexs_conflict: Dict = build_multiple_indexes(sub_conflict, [k for k in sub_conflict.keys()])
    # multiple_indexs_missing: Dict = build_multiple_indexes(sub_missing, [k for k in sub_missing.keys()])
    # multiple_indexs_origin: Dict = build_multiple_indexes(sub_origin, [k for k in sub_origin.keys()])
    # multiple_indexs_adv: Dict = build_multiple_indexes(sub_adv, [k for k in sub_adv.keys()])
    
    ### CBR CASE Matching output format
    """
    Dict
    key : question
    value: dict
    question: str
        origianl_case : list[dict]
            rank : int
            question : str
            context : str
            answer : str
        conflict_case : list[dict]
            rank : int
            question : str
            original_context : str
            context_with_random : str
            context_with_similar : str
            original_answer : str
            new_answer : str
        missing_case : list[dict]
            rank : int
            question : str
            original_context : str
            random_context : str
            similar_context : str
            hybrid_context : str
            original_answer : str
            new_answer : str
    """
    # conflict_case = match_case(qa_dataset, multiple_indexs_conflict, args)
    # missing_case = match_case(qa_dataset, multiple_indexs_missing, args)
    # original_case = match_case(qa_dataset, multiple_indexs_origin, args)
    # adv_case = match_case(qa_dataset, multiple_indexs_adv, args)
    # result = dict()
    # for q, c, m, o, adv in zip(qa_dataset["question"], conflict_case, missing_case, original_case, adv_case):
    #     result[q] = {"question":q, "conflict_case":c, "missing_case":m, "adv_case": adv, "original_case":o}
    
    result = dict()
    missings, originals, advs = [], [], []
    for row in missing_dataset:
        missings.append(make_row(row, "missing"))
    for row in original_dataset:
        originals.append(make_row(row, "original"))
    for row in adversary_dataset:
        advs.append(make_row(row, "adv"))
    for q in qa_dataset["question"]:
        random.seed(42)
        result[q] = {"question":q, "missing_case":random.sample(missings, args.num_cbr),
                     "adv_case":random.sample(advs, args.num_cbr),
                     "original_case":random.sample(originals, args.num_cbr)}
    json.dump(result, open(f"test_tqa.json", "w"))
    joblib.dump(result, f"/data/seongil/datasets/{args.output}.joblib")
    json.dump(vars(args), open(f"{args.output}.json", "w"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dataset", type=str, required=True, default="")
    parser.add_argument("--qa_split", type=str, required=False, default="train")
    parser.add_argument("--cbr_split", type=str, required=False, default="train")
    parser.add_argument("--num_cbr", type=int, required=False, default=5)
    parser.add_argument("--same_answer_filter", type=str2bool, required=False, default=False)
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--printing", type=str2bool, required=False, default=False)
    parser.add_argument("--output", type=str, required=True, default="test")
    args = parser.parse_args()
    main(args)
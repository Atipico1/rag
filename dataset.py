import json, joblib
import os
import random
from os import path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
from src.classes.prompt import PromptSet
from perturbations import add_adversarial_sentences_with_same_entity, add_conflicting_sentences, swap_context, swap_entities
from utils import check_same_answers_in, get_answer, get_question, normalize_answer, text_has_answer, update_context_with_substitution_string
from src.classes.cbr_data import NQ, NQExample, WikiContext
from src.classes.qadataset import QADataset, SquadDataset
from src.classes.qaexample import QAExample
from nltk import sent_tokenize
import numpy as np

def masking(output, model_type) -> str:
    if model_type=="spacy":
        if not len(output.ents):
            return output.text
        text_list =[]
        for d in output:
            if d.pos_ == "PUNCT":
                text_list.append("@"+d.text)
            elif d.pos_ == "AUX" and d.text == "'s":
                text_list.append("@"+d.text)
            else:
                text_list.append(d.text)

        for ent in output.ents:
            text_list[ent.start:ent.end] = ["[B]"]* (ent.end - ent.start)
            text_list[ent.start] = "[MASK]"
        return " ".join(text_list).replace(" [B]", "").replace(" @", "")
    elif model_type == "tner":
        batch_size = len(output["prediction"])
        for i in range(batch_size):
            for j in range(len(output["prediction"][i][j])):
                if output["prediction"][i][j] != "O":
                    output["input"][i][j] = "[MASK]"
        return [" ".join(res) for res in output["input"]]

def masking_entity(data: QADataset, model, model_type,  batch_size) -> QADataset:
    examples = [ex.query for ex in data.examples]
    result = []
    if model_type == "tner":
        for i in tqdm(range(0, len(examples), batch_size), desc="Masking..."):
            batch = examples[i:i+batch_size]
            batch_result = model.predict(batch)
            masked_quries = masking(batch_result, model_type)
            result.extend(masked_quries)
        assert len(result) == len(examples), "Length doesn't match"
    elif model_type == "spacy":
        for i in tqdm(range(0, len(examples), batch_size), desc="Masking..."):
            batch = examples[i:i+batch_size]
            batch_docs = list(model.pipe(batch, batch_size=batch_size))
            masked_quries = [masking(doc, model_type) for doc in batch_docs]
            result.extend(masked_quries)
        assert len(result) == len(examples), "Length doesn't match"
    for i in range(len(result)):
        data.examples[i].masked_query = result[i]
    return data

def remove_duplicate_queries(data: SquadDataset) -> SquadDataset:
    masked_queries: list[str] = [ex.masked_query for ex in data.examples]
    buffer = dict()
    result_idxs = []
    for idx, query in enumerate(masked_queries):
        if not buffer.get(query):
            buffer[query] = True
            result_idxs.append(idx)
    new_examples = []
    for idx, ex in enumerate(data.examples):
        if idx in result_idxs:
            new_examples.append(ex)
    data.examples = new_examples
    return data

def wh_filter(query: str, case: str) -> bool:
    wh_words = ["what", "when", "where", "which", "who", "whom", "whose", "why", "how"]
    normalized_query = normalize_answer(query.split()[0])
    normalized_case_q = normalize_answer(case.split()[0])
    if normalized_query in wh_words:
        if normalized_case_q == normalized_query:
            return True
        else:
            return False
    else:
        return True

def leakage_filtering(query: str, case_q: str) -> bool:
    if query == case_q:
        return True
    return False

def sent_embedding(data: QADataset, model_type:str,  model, tokenizer, batch_size):
    examples = data.examples
    result = []
    if model_type == "dpr":
        for i in tqdm(range(0, len(examples), batch_size), desc="Embedding..."):
            batch = examples[i:i+batch_size]
            sents = [b.masked_query for b in batch]
            output = tokenizer(sents, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [batch_size, hidden_dim]
            result.extend([emb for emb in embeddings])
    elif model_type == "roberta":
        for i in tqdm(range(0, len(examples), batch_size), desc="Embedding..."):
            batch = examples[i:i+batch_size]
            sents = [b.masked_query for b in batch]
            output = tokenizer(sents, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
            embeddings = model(**output).pooler_output.detach().cpu().numpy()
            result.extend([emb for emb in embeddings])
    else:
        raise NotImplementedError
    assert len(result) == len(examples), "Length doesn't match"
    for i in range(len(result)):
        data.examples[i].embedding = result[i]
    return data

def remove_duplicates(data: List[str]) -> List[int]:
    buffer = dict()
    result, result_idxs = [], []
    for idx, query in enumerate(data):
        if not buffer.get(query):
            buffer[query] = True
            result_idxs.append(idx)
            result.append(query)
    return result_idxs, result

def load_nq_train_data(batch_size, nlp_model, emb_model, tokenizer, file_path: str = "datasets/nq-train") -> NQ:
    if path.exists(file_path):
        print(f"Load nq-train from {file_path}")
        output = joblib.load(file_path)
        print("Load complete")
        return output
    else:
        with open(file_path+".json", "r") as f:
            data = json.load(f)
        data = list(filter(lambda x: any([ctx["has_answer"] for ctx in x["ctxs"]]), data))
        masked_embeddings, nqs, masked_queries, new_data = [], [], [], []
        for i in tqdm(range(0, len(data), batch_size), desc="NQ Train masking..."):
            batch = data[i:i+batch_size]
            _questions = [batch_data["question"] for batch_data in batch]
            batch_docs = list(nlp_model.pipe(_questions, batch_size=batch_size))
            masked_queries.extend([masking(doc) for doc in batch_docs])
        unique_indices, unique_masked_queries = remove_duplicates(masked_queries)
        for j, d in enumerate(data):
            if j in unique_indices:
                new_data.append(d)
        del data
        print("# of unique Queries:", len(unique_indices))
        for i in tqdm(range(0, len(new_data), batch_size), desc="NQ Train embedding..."):
            batch = new_data[i:i+batch_size]
            ids = [i+batch_i for batch_i in range(len(batch))]
            questions = [batch_data["question"] for batch_data in batch]
            answers = [batch_data["answers"] for batch_data in batch]
            contexts = [batch_data["ctxs"] for batch_data in batch]
            output = tokenizer(unique_masked_queries[i:i+batch_size],padding="max_length", truncation=True, max_length=256, return_tensors="pt").to("cuda")
            embeddings = emb_model(**output).pooler_output.detach().cpu().numpy()
            masked_embeddings = [emb for emb in embeddings]
            nqs.extend(
                [NQExample(id=id,
                           question=question,
                           answers=answer,
                           contexts=[WikiContext(id=ctx["id"], title=ctx["title"], text=ctx["text"], score=ctx["score"], has_answer=ctx["has_answer"]) for ctx in context],
                           masked_embedding=masked_embedding) for id, question, answer, context, masked_embedding in zip(ids, questions, answers, contexts, masked_embeddings)])
        print("# of CBR Examples:", len(nqs))
        output = NQ(dataset=nqs)
        joblib.dump(output, file_path)
    return output

def knowledge_separation(perturb_type: str, context: str) -> str:
    sent_list = sent_tokenize(context)
    mid = len(sent_list) // 2
    if perturb_type == "conflict":
        head, tail = " ".join(sent_list[:-1]), sent_list[-1]
    elif perturb_type == "adversarial":
        head, tail = " ".join(sent_list[:-3]), " ".join(sent_list[-3:])
    else:
        head, tail = " ".join(sent_list[:mid]), " ".join(sent_list[mid:])
    return head + "\n" + tail

def dataset_filter(nq_ex:QAExample, raw_data: Union[List[NQExample], List[QAExample]], args):
    """Filtering nq_train and squad dataset with nq_ex
    Args:
        nq_ex: NQExample
        nq_train: List[NQExample]
        squad: List[QAExample]
        args: dict"""
    if (args.filter_wh) and (not args.filter_same_answer):
        filtered = [ex for ex in raw_data if wh_filter(get_question(nq_ex), get_question(ex))]
        return filtered if len(filtered) > 0 else raw_data, None
    elif (args.filter_same_answer) and (not args.filter_wh):
        filtered = [ex for ex in raw_data if not check_same_answers_in(get_answer(ex), get_answer(nq_ex))]
        return filtered if len(filtered) > 0 else raw_data, None
    elif (args.filter_wh) and (args.filter_same_answer):
        filtered = [ex for ex in raw_data if (wh_filter(get_question(nq_ex), get_question(ex))) and (not check_same_answers_in(get_answer(ex), get_answer(nq_ex)))]
        return filtered if len(filtered) > 0 else raw_data, len([ex for ex in raw_data if check_same_answers_in(get_answer(ex), get_answer(nq_ex))])
    else:
        return raw_data, None

def context_embedding(data: QADataset, model, tokenizer, batch_size, save_path: str = "datasets/squad-context-embedding.joblib"):
    if os.path.exists(save_path):
        print(f"Load squad-context-embedding from {save_path}")
        result = joblib.load(save_path)
        print("Load complete")
        return result
    examples = data.examples
    unique_ctxs = list(set([ex.context for ex in examples if ex.context]))
    result = dict()
    for i in tqdm(range(0, len(unique_ctxs), batch_size), desc="Context Embedding..."):
        batch = unique_ctxs[i:i+batch_size]
        output = tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
        embeddings = model(**output).pooler_output.detach().cpu().numpy()
        [result.update({context:embedding/np.linalg.norm(embedding)})for context, embedding in zip(batch, embeddings)]
    joblib.dump(result, save_path)
    print("Context Embedding Save Complete")
    return result

def find_most_similar_context(data: QADataset, context_embeddings: Dict[str, np.ndarray]):
    """Find most similar context embedding with ex"""
    total = np.array([emb for _, emb in context_embeddings.items()])
    total_text = [text for text, _ in context_embeddings.items()]
    for ex in tqdm(data.examples, desc="Find most similar context"):
        query = context_embeddings[ex.context]
        scores = np.matmul(total, query.T)
        for i, idx in enumerate(list(np.argpartition(scores, -100)[-100:])):
            if i == 0:
                continue
            if not text_has_answer([answer.text for answer in ex.gold_answers], total_text[idx]):
                ex.metadata = {"most_similar_context":total_text[idx]}
                break
        # total = np.array([emb for ctx, emb in context_embeddings.items() if not text_has_answer([answer.text for answer in ex.gold_answers], ctx)])
        # total_text = [ctx for ctx, _ in context_embeddings.items() if not text_has_answer([answer.text for answer in ex.gold_answers], ctx)]
        # query = context_embeddings[ex.context]
        # scores = np.dot(total, query.T)
        # idx = np.argmax(scores)
        # ex.metadata = {"most_similar_context":total_text[idx]}

def make_conflict_context(data: List[PromptSet], num_conflict: int, strategy: str) -> List[PromptSet]:
    for d in data:
        if strategy == "random":
            rand_idx = random.sample(range(len(d.supports)), num_conflict)
            for idx, wiki in enumerate(d.supports):
                if idx in rand_idx:
                    wiki.text = update_context_with_substitution_string(wiki.text, d.answers, d.substitution)
        elif strategy == "topk":
            conflicted_contexts = sorted(d.supports, key=lambda x: x.score, reverse=True)[:num_conflict]
            original_contexts = sorted(d.supports, key=lambda x: x.score, reverse=True)[num_conflict:]
            for wiki in conflicted_contexts:
                wiki.text = update_context_with_substitution_string(wiki.text, d.answers, d.substitution)
            d.supports = conflicted_contexts + original_contexts
        d.answers = ["unanswerable"]
    return data

def make_perturb_testset(data: List[PromptSet], args: Dict) -> List[PromptSet]:
    if args["perturb_testset_op"] == "swap_context":
        has_no_answers: List[PromptSet] = list(filter(lambda x: not any([wiki.has_answer for wiki in x.supports[:args["num_wiki"]]]), data))
        has_answers: List[PromptSet] = list(filter(lambda x: any([wiki.has_answer for wiki in x.supports[:args["num_wiki"]]]), data))
        thres = int(len(data)*args["perturb_data_ratio"])
        for d in has_answers:
            if len(has_no_answers) >= thres:
                break
            d.supports = sorted([ctx for ctx in d.supports if not ctx.has_answer], key=lambda x: x.score, reverse=True)[:args["num_wiki"]]
            if len(d.supports) < args["num_wiki"]:
                continue
            else:
                has_no_answers.append(d)
        return data
    elif args["perturb_testset_op"] == "adversarial":
        pass
    elif args["perturb_testset_op"] == "confilct":
        thres = int(len(data)*args["perturb_data_ratio"])
        ctx_thres = int(args["num_wiki"]*args["perturb_context_ratio"])
        print(f"thres: {thres}, ctx_thres: {ctx_thres}")
        has_many_answers: List[PromptSet] = list(filter(lambda x: sum([wiki.has_answer for wiki in x.supports[:args["num_wiki"]]]) >= ctx_thres, data))
        has_few_answers: List[PromptSet] = list(filter(lambda x: sum([wiki.has_answer for wiki in x.supports[:args["num_wiki"]]]) < ctx_thres, data))
        print(f"has_many_answers: {len(has_many_answers)}, has_few_answers: {len(has_few_answers)}")
        while len(has_many_answers) < thres and len(has_few_answers) > 0:
            d = has_few_answers[-1]
            d.supports = sorted([ctx for ctx in d.supports if ctx.has_answer], key=lambda x: x.score, reverse=True)[:args["num_wiki"]]
            if len(d.supports) < args["num_wiki"]:
                has_few_answers.pop()
                continue
            else:
                has_many_answers.append(d)
                has_few_answers.pop()
        if len(has_many_answers) > thres:
            return make_conflict_context(
                data=has_many_answers[:thres],
                num_conflict=ctx_thres,
                strategy="topk") + has_many_answers[thres:] + has_few_answers
        else: # len(has_many_answers) == thres
            return make_conflict_context(
                data=has_many_answers[:thres],
                num_conflict=ctx_thres,
                strategy="topk") + has_few_answers
    elif args["perturb_testset_op"] == "entity_swap":
        pass
    else:
        pass
        
import argparse
import os
import json
import re
import string
import torch
from tqdm import tqdm
import openai
from collections import Counter
from typing import Union, Callable, List, Tuple, Dict
import numpy as np
import tiktoken
from src.classes.cbr_data import NQExample
from src.classes.qaexample import QAExample
from src.classes.answer import Answer
from src.classes.prompt import PromptSet
import wikipedia
import time
import random

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))

def normalize_answer(s: str):
    if not s:
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False

def get_answer_from_model_output(outputs, tokenizer, prompt):
    generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    generation_str = generation_str[len(prompt):]
    answer = generation_str.split("\n")[0]
    return answer, generation_str

def generate_answer_from_gpt(prompt: List[str], max_tokens: int):
    max_try = 0
    while max_try < 3:
        try:
            response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0
            )
            return [res["text"] for res in response["choices"]]
        except Exception as e:
            print(f"GPT API Error : {e}")
            max_try += 1
            time.sleep(3)
    print("GPT Failed to generate answer")
    return ""

def get_format_prompt(num_shot:int, examples_type:str) -> str:
    if examples_type == "zero":
        return ""
    format_path = f"datasets/format_prompt_{examples_type}.txt"
    output = ""
    with open(format_path, "r") as f:
        lines = f.readlines()
    for i in range(1, num_shot*3+1):
        output += lines[i-1]
        if i % 3 == 0:
            output += "\n"
    return output

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#TODO : query와 완전히 똑같은 few-shot은 제외하는 로직 추가
def find_topk(query: np.ndarray,
              key: List[np.ndarray],
              value: Union[List[QAExample], List[NQExample]],
              topk: int=10,
              filter_same_questions: bool = True,
              filtering_threshold: float = 0.95,
              random_selection: bool = False) -> List[int]:
    """
    query: [1, dim]
    key: [num_dataset, dim]
    output : [num_dataset, 1]
    res : list of integer, indices of topk
    """
    normalized_query = query / np.linalg.norm(query)
    normalized_key = np.array([k / np.linalg.norm(k) for k in key])
    if topk == 0:
        return []
    if len(value) <= topk:
        print("Few-shot examples are less than topk : ", len(value))
        return value
    output = np.matmul(normalized_key, normalized_query.T)
    if filter_same_questions:
        res = []
        res.extend(list(np.argpartition(output, -topk)[-1:]))
        i = 2
        while len(res) < topk:
            is_same = False
            for e in res:
                kth_largest = np.argpartition(output, -i)[-i]
                if cosine_similarity(normalized_key[kth_largest], normalized_key[e]) >= filtering_threshold:
                    is_same = True
            if not is_same:
                res.append(kth_largest)
            i += 1
        return [value[int(idx)] for idx in res]
    else:
        return [value[int(idx)] for idx in list(np.argpartition(output, -topk)[-topk:])]

def cal_num_tokens(encoder, input: str) -> int:
    return len(encoder.encode(input))

def find_sentence_with_span(span: Tuple[int, int], sentences: List[str]) -> Tuple[int, str]:
    cur_idx = 0
    for sent_idx, sent in enumerate(sentences):
        for char_idx, char in enumerate(list(sent)):
            if cur_idx >= span[0]:
                return sent_idx, sent
            cur_idx += 1
        cur_idx += 1

def extract_wiki_page(page_name: str):
    try:
        return wikipedia.page(page_name).content
    except:
        return None

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def check_answer(contexts: List[str], answers: List[str]) -> List[str]:
    output = []
    for context in contexts:
        has_answer = False
        for answer in answers:
            if answer in context.lower():
                has_answer = True
        if not has_answer and len(context.split()) < 60:
            output.append(context)
    return output
    
def merge_sentence(input: List[str], step: int=3) -> List[str]:
    output = []
    cnt = 0
    buffer = ""
    while input != []:
        buffer += input.pop() + " "
        cnt += 1
        if cnt == step:
            output.append(buffer.strip())
            cnt = 0
            buffer = ""
    if buffer:
        output.append(buffer.strip())
    return output

def make_adversarial(sent_idx: int, sents: List[str], sent_len: int, adversary: str, strategy: str) -> str:
    if strategy == "replace":
        mid = sent_len//2
        if sent_idx < mid:
            sents = sents[:mid]
            return " ".join(sents) + " " + adversary
        else:
            sents = sents[mid:]
            return adversary + " " + " ".join(sents)
    elif strategy == "add":
        return " ".join(sents) + " " + adversary
    else:
        raise NotImplementedError
    
def get_instruction(instruction_type: str) -> str:
    with open("/home/seongilpark/rag/datasets/instructions.txt", "r") as f:
        lines = f.readlines()
        instruct_dict = {line.split("||")[0]:line.split("||")[1].replace("\n", "")+"\n\n" for line in lines}
    return instruct_dict[instruction_type]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def check_same_answers_in(query: List[str] , key: List[str]) -> bool:
    """
    query: List[str]
    key: List[str]
    """
    for ans in query:
        for key_ans in key:
            if normalize_answer(ans) == normalize_answer(key_ans):
                return True
    return False

def get_answer(input: Union[NQExample, QAExample]) -> List[str]:
    if isinstance(input, NQExample):
        return input.answers
    elif isinstance(input, QAExample):
        return [gold.text for gold in input.gold_answers]

def get_answers(inputs: Union[List[NQExample], List[QAExample]]) -> List[List[str]]:
    if isinstance(inputs[0], NQExample):
        return [input.answers for input in inputs]
    elif isinstance(inputs[0], QAExample):
        return [[gold.text for gold in input.gold_answers] for input in inputs]

def get_question(input: Union[NQExample, QAExample]) -> str:
    if isinstance(input, NQExample):
        return input.question
    elif isinstance(input, QAExample):
        return input.query

def get_questions(inputs: Union[List[NQExample], List[QAExample]]) -> List[str]:
    if isinstance(inputs[0], NQExample):
        return [input.question for input in inputs]
    elif isinstance(inputs[0], QAExample):
        return [input.query for input in inputs]

def find_conflict_between_ctxs(ctxs: List[str], query: str, nlp, args: dict) -> bool:
    qa_input = [{"question":query, "context":ctx} for ctx in ctxs]
    results = nlp(qa_input)
    pred_answer = [result["answer"].lower().strip() for result in results]
    if len(set(pred_answer)) == 1:
        return False
    else:
        return True
    
def determine_perturbation_type(data: PromptSet, model:dict, args: dict):
    model["nli"]["model"].eval()
    query = data.query
    ctxs = [ctx.text for ctx in data.supports]
    features = model["nli"]["tokenizer"]([query]*len(ctxs), ctxs, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(args["device"])
    with torch.no_grad():
        scores = torch.nn.functional.sigmoid(model["nli"]["model"](**features).logits).detach().cpu().numpy()
        if np.where(scores > 0.5)[0].size == 0:
            return "no_relevant_ctx"
        else:
            relevant_ctxs = [ctxs[idx] for idx in np.where(scores > 0.5)[0]]
            if len(relevant_ctxs) <= 1:
                return "one_relevant_ctx"
            else:
                if find_conflict_between_ctxs(relevant_ctxs, query, model["nlp"], args):
                    return "conflict"
                else:
                    return "many_relevant_ctx"

def generate_answer(prompt, args):
    pass

def get_answer_from_lm(batch_prompt: List[str], args: dict) -> str:
    if args["lm"] == "gpt-3.5-turbo-instruct":
        return generate_answer_from_gpt(batch_prompt, args["max_tokens"])
    elif args["lm"] == "gpt-3.5-turbo-16k":
        return generate_answer_from_gpt(batch_prompt, args["max_tokens"])
    elif args["lm"] == "mistral-instruct":
        return generate_answer(batch_prompt, args)
    elif args["lm"] == "llama2-7b-chat":
        return generate_answer(batch_prompt, args)
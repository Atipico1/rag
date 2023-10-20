import os
import json
import re
import string
import torch
from tqdm import tqdm
import openai
from collections import Counter
from typing import Callable, List, Tuple
import numpy as np
import tiktoken
from src.classes.prompt import PromptSet



def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def normalize_answer(s: str):
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

def build_qa_prompt(example: PromptSet, format_prompts):
    q = normalize_question(example.query)
    if not example.fewshots:
        if example.num_docs == 0:
            #format_prompt = format_prompts["without_knowledge"]
            ex_prompt = f"Answer these questions:\nQuestion: when is the publishers clearing house sweepstakes drawing?\nAnswer: just after the Super Bowl\nQ: {q}\nA:"
        elif example.num_docs == 1:
            format_prompt = format_prompts
            knolwedge = example.supports[0].text
            ex_prompt = format_prompt + f"Knowledge: {knolwedge}\nQuestion: {q}\nAnswer:"
        else:
            format_prompt = format_prompts
            knolwedge = "\n".join([wiki.text for wiki in example.supports])
            ex_prompt = format_prompt + f"Knowledge: {knolwedge}\nQuestion: {q}\nAnswer:"
    else:
        fewshot = example.fewshots
        knolwedge = "\n".join([wiki.text for wiki in example.supports])
        ex_prompt = ""
        for shot in fewshot:
            shot_q, shot_a, shot_c = shot.question, shot.answer, shot.context
            ex_prompt += f"Knolwedge: {shot_c}\nQuestion: {shot_q}\nAnswer: {shot_a}\n"
        ex_prompt += f"Knolwedge: {knolwedge}\nQuestion: {q}\nAnswer:"
    return ex_prompt

def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match(prediction, ground_truth):
    # print(f"Pred : {normalize_answer(prediction)}")
    # print(f"ANS : {normalize_answer(ground_truth)}")
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_answer_from_model_output(outputs, tokenizer, prompt):
    generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    generation_str = generation_str[len(prompt):]
    answer = generation_str.split("\n")[0]
    return answer, generation_str

def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])

def get_answer_from_gpt(prompt: str, max_tokens: int):
    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0
    )
    return response["choices"][0]["text"]

def get_format_prompt(num_shot) -> str:
    output = ""
    with open("datasets/format_prompt.txt", "r") as f:
        lines = f.readlines()
    for i in range(num_shot*3):
        output += lines[i]
    return output

def evaluation(dataset: List[PromptSet],
               metadata: dict,
               output_dir="test-output/",
               num_shot=5,
               num_docs=10,
               max_tokens_to_generate=10,
               is_test=False,
               max_test=10):
    idx, num_correct, num_has_answer, num_tokens, is_recall = 0, 0, 0, 0, 0
    sample_prompt = None
    for_test = []
    encoder = tiktoken.get_encoding("cl100k_base")
    openai.api_key = "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"
    format_prompt = get_format_prompt(num_shot)
    if is_test:
        dataset = dataset[:max_test]
    for data in (tq := tqdm(dataset, desc=f"EM:  0.0%")):
        answers = data.answers
        prompt = build_qa_prompt(data, format_prompts=format_prompt)
        if idx == 0:
            print("Sample :\n", prompt, format_prompt)
            sample_prompt = prompt
        has_answer = text_has_answer(answers, prompt)
        prediction = get_answer_from_gpt(prompt, max_tokens_to_generate)
        is_correct = exact_match_score(prediction, answers, normalize_answer)
        idx += 1
        num_tokens += len(encoder.encode(prompt))
        num_correct += int(is_correct)
        num_has_answer += int(has_answer)
        is_recall += int(is_correct & has_answer)
        for_test.append({"Prompt":prompt,
                        "answers":answers,
                        "prediction":prediction,
                        "is_correct":is_correct})
        tq.set_description(f"EM: {num_correct / idx * 100:4.1f}%")

    with open(os.path.join(output_dir, "eval_test.json"), "w") as f:
        json.dump(for_test, f)
    em = round(num_correct / idx * 100, 3)
    has_answer = round(num_has_answer / idx * 100, 3)
    num_tokens = round(num_tokens / idx, 3)
    recall = round(is_recall / num_has_answer, 3)
    print(f"EM: {em:.1f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    if output_dir is not None:
        d = {"em": em, "has_answer(%)": has_answer, "recall":recall, "num_datasets": idx,
             "max_token":max_tokens_to_generate, "num_docs":num_docs, "num_tokens":num_tokens}
        [d.update({k:v}) for k, v in metadata.items()]
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)
    return d
def find_topk(query: np.ndarray, key: List[np.ndarray], topk: int=10) -> List[int]:
    """
    query: [dim]
    key: [num_dataset, dim]
    """
    if topk == 0:
        return []
    key_matrix = np.array(key)
    output = np.matmul(key_matrix, query.T)
    res = list(np.argpartition(output, -topk)[-topk:])
    return res

def cal_num_tokens(encoder, input: str) -> int:
    return len(encoder.encode(input))

def find_sentence_with_span(span: Tuple[int, int], sentences: List[str]) -> Tuple[int, str]:
    cur_idx = 0
    for sent_idx, sent in enumerate(sentences):
        for char_idx, char in enumerate(list(sent)):
            if span[0] == cur_idx:
                return sent_idx, sent
            cur_idx += 1

import os
import json
import re
import string
import torch
from tqdm import tqdm
import openai
from collections import Counter
from typing import Callable

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

def build_qa_prompt(example, num_docs=1):
    if num_docs == 0:
        question_text = normalize_question(example["question"])
        ex_prompt = f"Answer these questions:\nQ: when is the publishers clearing house sweepstakes drawing?\nA: just after the Super Bowl\nQ: {question_text}\nA:"
    elif num_docs == 1:
        q = normalize_question(example["question"])
        title = example['ctxs'][0]['title']
        text = example['ctxs'][0]['text']
        ex_prompt = f"{title}\n\n{text}\n\nBased on this text, answer these questions:\nQ: {q}\nA:"
    else:
        q = normalize_question(example["question"])
        docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:num_docs]])
        ex_prompt = f"{docs_text}\n\nBased on these texts, answer these questions:\nQ: {q}\nA:"

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

def evaluate_gpt(dataset, num_docs=0, output_dir=None, max_tokens_to_generate=10, is_test=False, max_test=10):
    idx = 0
    num_correct = 0
    num_has_answer = 0
    sample_prompt = None
    for_test = []
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if is_test:
        dataset = dataset[:max_test]
    for data in (tq := tqdm(dataset, desc=f"EM:  0.0%")):
        answers = data["answers"]
        prompt = build_qa_prompt(data, num_docs=num_docs)
        if idx == 0:
            print("Sample :\n", prompt)
            sample_prompt = prompt
        has_answer = text_has_answer(answers, prompt)
        prediction = get_answer_from_gpt(prompt, max_tokens_to_generate)
        #is_correct = text_has_answer(answers, prompt)
        is_correct = exact_match_score(prediction, answers, normalize_answer)
        idx += 1
        if is_correct:
            num_correct += 1
        if has_answer:
            num_has_answer += 1
        if is_test:
            for_test.append({"Prompt":prompt,
                             "answers":answers,
                             "prediction":prediction,
                             "is_correct":is_correct})
        tq.set_description(f"EM: {num_correct / idx * 100:4.1f}%")

    if is_test:
        with open(os.path.join(output_dir, "eval_test.json"), "w") as f:
            json.dump(for_test, f)
    em = num_correct / idx * 100
    has_answer = num_has_answer / idx * 100
    print(f"EM: {em:.1f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    if output_dir is not None:
        d = {"em": em, "has_answer(%)": has_answer, "num_datasets": idx, "max_token":max_tokens_to_generate, "num_docs":num_docs}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)

def evaluate_dataset(
        model, tokenizer, device, eval_dataset, max_length, num_docs=0, output_dir=None, max_tokens_to_generate=10
):
    idx = 0
    num_correct = 0
    num_has_answer = 0
    num_too_long = 0
    sample_prompt = None
    for ex in (tq := tqdm(eval_dataset, desc=f"EM:  0.0%")):
        answers = ex["answers"]
        prompt = build_qa_prompt(ex, num_docs=num_docs)
        if idx == 0:
            sample_prompt = prompt
        has_answer = text_has_answer(answers, prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[-1] > max_length - max_tokens_to_generate:
            num_too_long += 1
            input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=max_tokens_to_generate)

        prediction, generation = get_answer_from_model_output(outputs, tokenizer, prompt)
        is_correct = any([exact_match(prediction, answer) for answer in answers])

        idx += 1
        if is_correct:
            num_correct += 1
        if has_answer:
            num_has_answer += 1
        tq.set_description(f"EM: {num_correct / idx * 100:4.1f}%")

    em = num_correct / idx * 100
    has_answer = num_has_answer / idx * 100
    print(f"EM: {em:.1f}%")
    print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
    if output_dir is not None:
        d = {"em": em, "has_answer": has_answer, "num_examples": idx, "too_long": num_too_long}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)
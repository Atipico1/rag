import torch
import json
import pandas as pd
from tqdm.auto import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoder
import argparse
from dataset import masking
from utils import str2bool
import wandb
import numpy as np
import spacy
import os
from copy import deepcopy
RELEVANT_TEMPLATE = 'Given the following text by a user, extract the part that is related and useful, so that using that text alone would be good context for providing an accurate and correct answer to the question portion of the text. Please include the actual question or query that the user is asking. Separate this into two categories labeled with ”Context text related to the question (includes all content except unrelated sentences):” and ”Detailed question:”. Do not use list.\n\nText by User: [ORIGINAL INPUT PROMPT]'
# [ORIGINAL INPUT PROMPT] : is the placeholder for the original input prompt. Please do not remove it.
UNBIASED_TEMPLATE = '[INPUT CONTEXT]\n\nAnswer in an unbiased way.'
UNBIASED_TEMPLATE_WITH_FORMAT = '[INPUT CONTEXT]\n\nAnswer in an unbiased way. Please separate into two categories labeled with ”Solution:” and ”Final answer (in numbers):”'
ZERO_SHOT_TEMPLATE = 'Solve the following math problem. Please separate into two categories labeled with ”Solution:” and ”Final answer(in numbers):”\n\nProblem: [MATHPROBLEM]'
INSTRUCTED_TEMPLATE = 'Solve the following math problem. If there is part that is irrelevant and not useful for providing an accurate and correct answer to the question portion of the text, you may ignore that part. Please separate into two categories labeled with ”Solution:” and ”Final answer(in numbers):”\nProblem: [MATHPROBLEM]'

## ORCA
ORCA_SYSTEM_MESSAGE = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."

## MODEL
TOKEN_MAP = {"meta-llama/Llama-2-13b-chat-hf": "[/INST]",
             "microsoft/Orca-2-13b": "<|im_start|> assistant"}

def make_prompt(input_text:str, args):
    if "chat" in args.model:
        template = f"<s>[INST] <<SYS>You are a helpful assitant.<</SYS>>>\n\n{input_text} [/INST]"
        return template
    elif "Orca" in args.model:
        template = f"<|im_start|>system\n{ORCA_SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant"
        return template
        

def make_case_prompt(case_questions: List[str], case_answers: List[str], query: str):
    template = "<s>[INST] <<SYS>You are a helpful assitant.<</SYS>>>\n\n[INPUT] [/INST]"
    prompt = ""
    for idx, (case_question, case_answer) in enumerate(zip(case_questions, case_answers)):
        prompt += f"Example{idx+1}:\nProblem:: {case_question}\nFinal answer (in numbers): {case_answer}\n\n"
    prompt += f"Based on the above examples, solve the following math problem in an unbiased way. If there is part that is irrelevant and not useful for providing an accurate and correct answer to the question portion of the text, you may ignore that part. Please separate into two categories labeled with ”Solution:” and ”Final answer (in numbers):”\n\nProblem: {query}"
    return template.replace("[INPUT]", prompt)

def parse_outputs(raw_outputs: List[str]):
    parsed_outputs = []
    for raw_output in raw_outputs:
        if "Final answer (in numbers):" in raw_output:
            parsed_output = raw_output.split("Final answer (in numbers):")[1].strip()
        elif "Final answer:" in raw_output:
            parsed_output = raw_output.split("Final answer:")[1].strip()
        elif "Final answer(in numbers):" in raw_output:
            parsed_output = raw_output.split("Final answer(in numbers):")[1].strip()
        else:
            parsed_output = "parse error"
        parsed_outputs.append(parsed_output)
    return parsed_outputs

def parse_tokens(outputs: List[str], batch_prompts: List[str]):
    result = []
    for output, prompt in zip(outputs, batch_prompts):
        output = output[:len(prompt)]
        result.append(output)
    return result

def S2A(questions: List[str], tokenizer, model, args):
    real_preds, first_prompts, second_prompts, raw_outputs = [], [], [], []
    with torch.no_grad():
        if args.batch_size > 1:
            for i in tqdm(range(0, len(questions), args.batch_size), desc="S2A..."):
                batch = questions[i:i+args.batch_size]
                s2a_prompts = [make_prompt(RELEVANT_TEMPLATE.replace("[ORIGINAL INPUT PROMPT]", question), args) for question in batch]
                encoded_prompt = tokenizer(s2a_prompts, padding="longest", return_tensors="pt").to("cuda")
                first = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
                first_prompts.extend(s2a_prompts)
                pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(first, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
                real_prompt = [make_prompt(UNBIASED_TEMPLATE_WITH_FORMAT.replace("[INPUT CONTEXT]", p), args) for p in pred]
                second_prompts.extend(real_prompt)
                encoded_prompt = tokenizer(real_prompt, padding="longest", return_tensors="pt").to("cuda")
                second = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
                pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(second, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
                raw_outputs.extend(pred)
                real_pred = parse_outputs(pred)
                real_preds.extend(real_pred)
        else:
            for question in tqdm(questions, desc="S2A..."):
                s2a_prompt = make_prompt(RELEVANT_TEMPLATE.replace("[ORIGINAL INPUT PROMPT]", question), args)
                inputs = tokenizer(s2a_prompt, return_tensors="pt").to("cuda")
                first_output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, use_cache=True)
                first_prompts.extend(tokenizer.batch_decode(first_output_ids, skip_special_tokens=True))
                sequence_length = inputs["input_ids"].shape[1]
                new_output_ids = first_output_ids[:, sequence_length:]
                first_answer = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0]
                
                second_prompt = make_prompt(UNBIASED_TEMPLATE_WITH_FORMAT.replace("[INPUT CONTEXT]", first_answer), args)
                inputs = tokenizer(second_prompt, return_tensors="pt").to("cuda")
                second_output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, use_cache=True)
                second_prompts.extend(tokenizer.batch_decode(second_output_ids, skip_special_tokens=True))
                sequence_length = inputs["input_ids"].shape[1]
                new_output_ids = second_output_ids[:, sequence_length:]
                second_answer = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0]
                raw_outputs.append(second_answer)
                pred = parse_outputs([second_answer])
                real_preds.extend(pred)
    return real_preds, first_prompts, second_prompts, raw_outputs

def query_masking(nlp, dataset: pd.DataFrame) -> pd.DataFrame:
    questions = dataset["new_question"].tolist()
    result = []
    for i in tqdm(range(0, len(questions), 2000), desc="Masking..."):
        batch = questions[i:i+2000]
        batch_docs = list(nlp.pipe(batch, batch_size=2000))
        masked_quries = [masking(doc, "spacy") for doc in batch_docs]
        result.extend(masked_quries)
    assert len(result) == len(questions), "Length doesn't match"
    dataset["masked_query"] = result
    return dataset

def query_embedding(model, tokenizer, dataset: pd.DataFrame):
    queries = dataset["masked_query"].tolist()
    result = []
    for i in tqdm(range(0, len(queries), 1000), desc="Embedding..."):
        batch = queries[i:i+1000]
        output = tokenizer(batch, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [args.batch_size, hidden_dim]
        result.extend([emb for emb in embeddings])
    assert len(result) == len(queries), "Length doesn't match"
    dataset["query_embedding"] = result
    return dataset

def case_match(query_set: pd.DataFrame, case_set: pd.DataFrame, args) -> str:
    if os.path.exists(f"{args.dataset_size}_{args.subset}_query") and os.path.exists(f"{args.dataset_size}_{args.subset}_case"):
        query_set = pd.read_pickle(f"{args.dataset_size}_{args.subset}_query")
        case_set = pd.read_pickle(f"{args.dataset_size}_{args.subset}_case")
        print("Loading done!")
    else:
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        query_set, case_set = query_masking(nlp, query_set), query_masking(nlp, case_set)
        query_set, case_set = query_embedding(model, tokenizer, query_set), query_embedding(model, tokenizer, case_set)
        query_set.to_pickle(f"{args.dataset_size}_{args.subset}_query")
        case_set.to_pickle(f"{args.dataset_size}_{args.subset}_case")
        print("Embedding and Saving done!")
    result = []
    for idx, query in tqdm(enumerate(query_set["query_embedding"].tolist()), desc="Case Matching..."):
        query = np.array(query) # [1, hidden_dim]
        query = query / np.linalg.norm(query)
        case_embeddings = np.array(case_set[case_set["original_question"]!=query_set.iloc[idx]["original_question"]]["query_embedding"].tolist()) # [num_cases w/ duplicated queries, hidden_dim]
        similarities = np.matmul(case_embeddings, query.T) # [num_cases, 1]
        topk_idxes = list(np.argpartition(similarities, -args.num_cases)[-args.num_cases:])  
        topk_questions = case_set[case_set["original_question"]!=query_set.iloc[idx]["original_question"]].iloc[topk_idxes]["new_question"].tolist()
        topk_answers = case_set[case_set["original_question"]!=query_set.iloc[idx]["original_question"]].iloc[topk_idxes]["answer"].tolist()
        case_based_prompt = make_case_prompt(topk_questions, topk_answers, query_set.iloc[idx]["new_question"])
        result.append(case_based_prompt)
    return result

def CBR(prompts: List[str], tokenizer, model, args):
    preds, raw_outputs = [], []
    with torch.no_grad():
        if args.batch_size > 1:
            for i in tqdm(range(0, len(prompts), args.batch_size), desc="CBR..."):
                batch = prompts[i:i+args.batch_size]
                encoded_prompt = tokenizer(batch, padding="longest", return_tensors="pt").to("cuda")
                pred = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
                pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(pred, skip_special_tokens=True)]
                raw_outputs.extend(pred)
                pred = parse_outputs(pred)
                preds.extend(pred)
        else:
            for prompt in tqdm(prompts, desc="CBR..."):
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, use_cache=True)
                raw_outputs.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
                sequence_length = inputs["input_ids"].shape[1]
                new_output_ids = output_ids[:, sequence_length:]
                answer = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0]
                pred = parse_outputs([answer])
                preds.extend(pred)
    return preds, raw_outputs

def baseline(questions: List[str], tokenizer, model, args):
    preds, prompts, raw_outputs = [], [], []
    with torch.no_grad():
        if args.batch_size > 1:
            for i in tqdm(range(0, len(questions), args.batch_size), desc="Baseline..."):
                batch = questions[i:i+args.batch_size]
                baseline_prompt = [make_prompt(ZERO_SHOT_TEMPLATE.replace("[MATHPROBLEM]", question), args) for question in batch]
                encoded_prompt = tokenizer(baseline_prompt, padding="longest", return_tensors="pt").to("cuda")
                pred = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
                pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(pred, skip_special_tokens=True)]
                raw_outputs.extend(pred)
                pred = parse_outputs(pred)
                preds.extend(pred)
                print("Baseline: ", pred)
                prompts.extend(baseline_prompt)
        else:
            for question in tqdm(questions, desc="Baseline..."):
                prompt = make_prompt(ZERO_SHOT_TEMPLATE.replace("[MATHPROBLEM]", question), args)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, use_cache=True)
                raw_outputs.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
                sequence_length = inputs["input_ids"].shape[1]
                new_output_ids = output_ids[:, sequence_length:]
                answer = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0]
                pred = parse_outputs([answer])
                preds.extend(pred)
                prompts.append(prompt)
    return preds, prompts, raw_outputs

def instruction(questions: List[str], tokenizer, model, args):
    preds, prompts, raw_outputs = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc="Instruction..."):
            batch = questions[i:i+args.batch_size]
            batch_prompt = [make_prompt(INSTRUCTED_TEMPLATE.replace("[MATHPROBLEM]", question), args) for question in batch]
            encoded_prompt = tokenizer(batch_prompt, padding="longest", return_tensors="pt").to("cuda")
            pred = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
            pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(pred, skip_special_tokens=True)]
            raw_outputs.extend(pred)
            pred = parse_outputs(pred)
            preds.extend(pred)
            prompts.extend(baseline_prompt)
    return preds, prompts, raw_outputs

def oracle(questions: str, tokenizer, model, args):
    preds, prompts, raw_outputs = [], [], []
    with torch.no_grad():
        if args.batch_size > 1:
            for i in tqdm(range(0, len(questions), args.batch_size), desc="Oracle..."):
                batch = questions[i:i+args.batch_size]
                batch_prompt = [make_prompt(ZERO_SHOT_TEMPLATE.replace("[MATHPROBLEM]", question), args) for question in batch]
                encoded_prompt = tokenizer(batch_prompt, padding="longest", return_tensors="pt").to("cuda")
                output = model.generate(**encoded_prompt, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)
                #pred = parse_tokens(tokenizer.batch_decode(output, skip_special_tokens=True), batch_prompt)
                pred = [p.split(TOKEN_MAP[args.model])[-1].strip() for p in tokenizer.batch_decode(output, skip_special_tokens=True)]
                raw_outputs.extend(pred)
                pred = parse_outputs(pred)
                print("Final Output: ", pred)
                preds.extend(pred)
                prompts.extend(batch_prompt)
        else:
            for question in tqdm(questions, desc="Oracle..."):
                prompt = make_prompt(ZERO_SHOT_TEMPLATE.replace("[MATHPROBLEM]", question), args)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, use_cache=True)
                raw_outputs.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
                sequence_length = inputs["input_ids"].shape[1]
                new_output_ids = output_ids[:, sequence_length:]
                answer = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0]
                pred = parse_outputs([answer])
                preds.extend(pred)
                prompts.append(prompt)
    return preds, prompts, raw_outputs

def load_dataset(args):
    with open("/home/seongilpark/rag/GSM-IC_2step.json", "r") as f:
        df = pd.DataFrame(json.load(f))
    if args.subset == "in_topic":
        df = df[df["sentence_label"] == "in_topic"]
    if args.test:
        sample_data = df.sample(4, random_state=42)
        remain_data = df.drop(sample_data.index)
        return sample_data, remain_data
    else:
        sample_data = df.sample(args.dataset_size, random_state=42)
        remain_data = df.drop(sample_data.index)
        return sample_data, remain_data

def cal_match_acc(answer: str, pred: str):
    answer = answer.lower().strip()
    pred = pred.lower().strip()
    if answer in pred:
        return True
    else:
        return False

def main(args):
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    if "chat" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                    trust_remote_code=True,
                                                    torch_dtype = torch.bfloat16,
                                                    device_map="auto",
                                                    use_flash_attention_2=True)
    elif "Orca" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  model_max_length=4096,
                                                  padding_side="right",
                                                  use_fast=False,
                                                  add_special_tokens=False)
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                    trust_remote_code=True,
                                                    torch_dtype = torch.bfloat16,
                                                    device_map="auto",
                                                    use_flash_attention_2=True)
    tokenizer.pad_token = tokenizer.eos_token
    df, remain = load_dataset(args)
    print("Dataset loaded successfully ... Length of dataset: ", len(df))
    output = dict()
    if args.task in ["baseline", "cbr", "s2a"]:
        questions = df["new_question"].tolist()
    else:
        questions = df["original_question"].tolist()
    if args.task == "cbr":
        prompts = case_match(df, remain, args)
        preds, raw_outputs = CBR(prompts, tokenizer, model, args)
        output.update({"preds":preds, "prompts":prompts, "raw_outputs":raw_outputs})
    elif args.task == "s2a":
        preds, s2a_prompts, output_prompts, raw_outputs = S2A(questions, tokenizer, model, args)
        output.update({"preds":preds, "s2a_prompts":s2a_prompts, "output_prompts":output_prompts, "raw_outputs":raw_outputs})
    elif args.task == "baseline":
        preds, prompts, raw_outputs = baseline(questions, tokenizer, model, args)
        output.update({"preds":preds, "prompts":prompts, "raw_outputs":raw_outputs})
    elif args.task == "instruction":
        preds, prompts, raw_outputs = instruction(questions, tokenizer, model, args)
        output.update({"preds":preds, "prompts":prompts, "raw_outputs":raw_outputs})
    else:
        preds, prompts, raw_outputs = oracle(questions, tokenizer, model, args)
        output.update({"preds":preds, "prompts":prompts, "raw_outputs":raw_outputs})
    answers = df["answer"].tolist()
    is_accurates = [cal_match_acc(answer, pred) for answer, pred in zip(answers, preds)]
    output.update({"is_accurate":is_accurates, "answers":answers})
    result = pd.DataFrame(output)
    wandb.init(project="CBR-RAG-ADV", name=f"{'test_' if args.test else ''}{str(args.dataset_size)}_{args.task}_{args.subset}{'_case:'+str(args.num_cases) if args.task == 'cbr' else ''}_{args.model.split('/')[-1]}")
    tbl_result = wandb.Table(dataframe=result)
    wandb.log({"accuracy": round(result["is_accurate"].mean()*100,3), "parse_error": result[result["preds"] == "parse error"].shape[0]})
    wandb.log({"result":tbl_result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="meta-llama/Llama-2-13b-chat-hf", choices=["microsoft/Orca-2-13b", "meta-llama/Llama-2-13b-chat-hf"])
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--dataset_size", type=int, required=False, default=200)
    parser.add_argument("--num_cases", type=int, required=False, default=1)
    parser.add_argument("--task", type=str, required=False, default="all", choices=["all", "s2a", "cbr", "baseline", "oracle", "instruction"])
    parser.add_argument("--subset", type=str, required=False, default="in_topic", choices=["in_topic", "random"])
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    args = parser.parse_args()
    main(args)
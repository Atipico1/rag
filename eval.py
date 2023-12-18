import argparse
import os
import tiktoken
import wandb
from openai import OpenAI
from metrics import cal_metrics, exact_match_score
from train import preprocess_dataset
from utils import generate_answer_from_gpt, generate_answer_from_gpt_ensemble, get_instruction, normalize_answer, str2bool, text_has_answer
from transformers import pipeline
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset
from peft import PeftModel, PeftConfig
import pandas as pd
import joblib
from nltk import sent_tokenize
from typing import List, Dict
from math import exp
from collections import defaultdict
import random

MODEL_COMMIT_MAP = {
    "Seongill/nq_checkpoints_2": "3c1549b",
    "Seongill/nq_mrc_checkpoints": "58041cf2982005d78429642c218cf31b0fb8eca8",
    "Seongill/nq_checkpoints_cbr": "65b5efd726d85cfb5098140ddbc7ce11bfded29f",
    "Seongill/nq_mrc_cbr_checkpoints": "640f4f4b909c3dab5d55563b1d639be9ae188838"
}

def make_short_context(cbr_case: Dict, args):
    """
    "question":row["question"],
    "original_context":row["context"],
    "context_with_random_answer":row["random_answer"],
    "context_with_similar_answer":row["similar_answer"],
    "context_with_random": update_context_with_substitution_string(row["rewritten_context"], row["answers"]["text"], row["random_answer"]),
    "context_with_similar": update_context_with_substitution_string(row["rewritten_context"], row["answers"]["text"], row["similar_answer"]),
    "new_answer": "conflict",
    "original_answer": row["answers"]["text"]}
    """
    original_context = cbr_case["original_context"]
    original_sents = sent_tokenize(original_context)
    original_answer = cbr_case["original_answer"][0]
    
    explanation = ""
    for idx, sent in enumerate(original_sents):
        if original_answer in sent:
            answer_sent_idx = idx
    explanation += f"The first document states '{original_sents[answer_sent_idx]}'. "
    start_idx, end_idx = max(0, answer_sent_idx-3), min(len(original_sents), answer_sent_idx+4)
    short_context = " ".join(original_sents[start_idx:end_idx])
    if args.doc_numbering:
        short_context = f"Doc 0: " + short_context
    rewritten_context = cbr_case["context_with_similar"]
    rewritten_sents = sent_tokenize(rewritten_context)
    similar_answer = cbr_case["context_with_similar_answer"]
    for idx, sent in enumerate(rewritten_sents):
        if similar_answer in sent:
            answer_sent_idx = idx
    explanation += f"The second document states '{rewritten_sents[answer_sent_idx]}'. So there is a conflict between documents."
    start_idx, end_idx = max(0, answer_sent_idx-3), min(len(rewritten_sents), answer_sent_idx+4)
    if args.doc_numbering:
        short_context += "\n" + "Doc 1: " + " ".join(rewritten_sents[start_idx:end_idx])
    else:
        short_context += "\n" + " ".join(rewritten_sents[start_idx:end_idx])
    return short_context, explanation
   
def bulid_cbr_prompt(args, question, q_to_cbr):
    #TODO 경로명 설정
    output = ""
        
    if args.cbr:
        for idx, cbr_case in enumerate(q_to_cbr["original_case"][:args.num_cbr_cases]):
            if args.doc_numbering:
                output += f"Knowledge:\nDoc 0: {cbr_case['context']}\nQ: {cbr_case['question']}\nA: {cbr_case['answer']}\n\n"
            else:
                output += f"Knowledge: {cbr_case['context']}\nQ: {cbr_case['question']}\nA: {cbr_case['answer']}\n\n"
    if args.cbr_perturb_type == "conflict":
        for cbr_case in q_to_cbr["conflict_case"][:args.num_perturb_cases]:
            if args.short_context:
                short_context, explanation = make_short_context(cbr_case, args)
                if args.explain:
                    cbr_case["new_answer"] = "conflict. " + explanation
                if args.prepend:
                    if args.doc_numbering:
                        output = f"Knowledge:\n{short_context}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n" + output
                    else:
                        output = f"Knowledge: {short_context}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n" + output
                else:
                    if args.doc_numbering:
                        output += f"Knowledge:\n{short_context}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n"
                    else:
                        output += f"Knowledge: {short_context}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n"
            else:
                if args.prepend:
                    output = f"Knowledge: {short_context}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n" + output
                else:
                    output += f"Knowledge: {cbr_case['original_context']}\n{cbr_case['context_with_similar']}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n"
        return output
    elif args.cbr_perturb_type == "missing":
        for cbr_case in q_to_cbr["missing_case"][:args.num_perturb_cases]:
            output += f"Knowledge: {cbr_case['hybrid_context']}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n"
        return output
    elif args.cbr_perturb_type == "adv":
        for cbr_case in q_to_cbr["adv_case"][:args.num_perturb_cases]:
            output += f"Knowledge: {cbr_case['new_context']}\nQ: {cbr_case['question']}\nA: {cbr_case['answer']}\n\n"
        return output
    elif args.cbr_perturb_type == "both":
        for cbr_case in q_to_cbr["missing_case"][:args.num_perturb_cases]:
            output += f"Knowledge: {cbr_case['hybrid_context']}\nQ: {cbr_case['question']}\nA: {cbr_case['new_answer']}\n\n"
        for cbr_case in q_to_cbr["adv_case"][:args.num_perturb_cases]:
            output += f"Knowledge: {cbr_case['new_context']}\nQ: {cbr_case['question']}\nA: {cbr_case['answer']}\n\n"
        return output
    else:
        return output

def make_prompt_for_gpt(dataset: Dataset, args, q_to_cbrs=None):
    def sub_fn(ex):
        prompt = get_instruction(args.instruction)
        contexts = "\n".join([ex["text"] for ex in ex["ctxs"][:args.num_contexts]])
        if args.cbr: 
            prompt += bulid_cbr_prompt(args, ex['question'], q_to_cbrs[ex['question']])
        if args.doc_numbering:
            prompt += "Knowledge:\n" + "\n".join([f'Doc {i}: {content}' for i, content in enumerate(contexts.split("\n"))]) + "\n"
        else:
            prompt += "Knowledge: " + contexts + "\n"
        prompt += f"Q: {ex['question']}\n"
        prompt += "A:"
        return {"prompt":prompt}
    return dataset.map(sub_fn)

def find_max_value_key(input_dict):
    max_key = max(input_dict, key=input_dict.get)
    return max_key

def reduce_prompts(responses, weights: List[float]):
    max_len = max([len(response.logprobs.tokens) for response in responses])
    weights = [exp(weight) for weight in weights]
    weights = [weight/sum(weights) for weight in weights]
    [response.logprobs.top_logprobs.extend([{"<|endoftext|>":0}]*(max_len-len(response.logprobs.tokens))) for response in responses]
    time_step_logprobs = {k:defaultdict(float) for k in range(max_len)}
    for idx, response in enumerate(responses):
        for t, item in enumerate(response.logprobs.top_logprobs):
            for k, v in item.items():
                item[k] = exp(v)*weights[idx]
                time_step_logprobs[t][k] += item[k]
    tokens = []
    for t in range(max_len):
        max_key = find_max_value_key(time_step_logprobs[t])
        if max_key.strip() == "<|endoftext|>":
            break
        tokens.append(max_key)
    return "".join(tokens)

def make_random_cbr(q_to_cbrs, args):
    original_cases, conflict_cases, missing_cases, adv_cases = [], [], [], []
    for question, cbrs in q_to_cbrs.items():
        for key, cbr in cbrs.items():
            if key == "original_case": original_cases.append(cbr)
            elif key == "conflict_case": conflict_cases.append(cbr)
            elif key == "missing_case": missing_cases.append(cbr)
            elif key == "adv_case": adv_cases.append(cbr)
    new_cbrs = {}
    for question, cbrs in q_to_cbrs.items():
        new_cbr = dict()
        for key, cbr in cbrs.items():
            if key == "original_case":
                random_sample = random.choice(original_cases)
                while random_sample == q_to_cbrs[question]["original_case"]:
                    random_sample = random.choice(original_cases)
                new_cbr[key] = random_sample
            elif key == "conflict_case":
                random_sample = random.choice(conflict_cases)
                while random_sample == q_to_cbrs[question]["conflict_case"]:
                    random_sample = random.choice(conflict_cases)
                new_cbr[key] = random_sample
            elif key == "missing_case":
                random_sample = random.choice(missing_cases)
                while random_sample == q_to_cbrs[question]["missing_case"]:
                    random_sample = random.choice(missing_cases)
                new_cbr[key] = random_sample
            elif key == "adv_case":
                random_sample = random.choice(adv_cases)
                while random_sample == q_to_cbrs[question]["adv_case"]:
                    random_sample = random.choice(adv_cases)
                new_cbr[key] = random_sample
        new_cbrs[question] = new_cbr
    assert len(new_cbrs) == len(q_to_cbrs)
    return new_cbrs

def ensemble(dataset: Dataset, args, q_to_cbrs=None):
    prompts, preds = [], []
    client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    if args.random_cbr:
        q_to_cbrs = make_random_cbr(q_to_cbrs, args)
    for row in tqdm(dataset, desc="GPT Generating..."):
        ensemble_prompt, weights = [], []
        for ctx in row["ctxs"][:args.num_contexts]:
            prompt = get_instruction(args.instruction)
            if args.cbr:
                prompt += bulid_cbr_prompt(args, row['question'], q_to_cbrs[row['question']])
            prompt += f"Knowledge: {ctx['text']}\nQ: {row['question']}\nA:"
            ensemble_prompt.append(prompt)
            weights.append(ctx["score"])
        #prompts.append(ensemble_prompt)
        prompts.append("\n\n".join(ensemble_prompt))
        ensemble_preds = generate_answer_from_gpt_ensemble(ensemble_prompt, client, args.max_new_tokens)
        pred = reduce_prompts(ensemble_preds, weights)
        preds.append(pred)
    return prompts, preds

def parse_answers(preds: List[str], args):
    if args.explain:
        for idx, pred in enumerate(preds):
            if "conflict" in pred.lower().strip():
                preds[idx] = "conflict"
    return preds

def make_answer_for_gpt(dataset: Dataset, args):
    def sub_fn(ex):
        if "conflict" in args.dataset_name:
            return {"new_answers": ["conflict"] if ex["is_conflict"] else ex["answers"]}
        elif "missing" in args.dataset_name:
            return {"new_answers": ["unanswerable"] if not ex["has_answer"] else ex["answers"]}
        else:
            return {"new_answers":ex["answers"]}
    return dataset.map(sub_fn)
        
def main(args):
    if "gpt" in args.model:
        client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
        dataset: Dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        if args.test:
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.select(range(args.test_size))
            # if args.cbr_perturb_type == "adv":
            #     advs = dataset["num_advs"]
            #     print("ADV Ratio: ", len([i for i in advs if i != 0])/len(dataset))
        if args.cbr:
            q_to_cbrs = joblib.load(f"/data/seongil/datasets/{args.q_to_cbrs}.joblib")
        else:
            q_to_cbrs = None
        if args.ensemble:
            dataset = make_answer_for_gpt(dataset, args)
            prompts, preds = ensemble(dataset, args, q_to_cbrs)
            dataset = dataset.add_column("prompt", prompts)
        else:
            if args.custom_questions:
                if args.filter_option == "include":
                    custom_questions = joblib.load(args.custom_questions) # List[str]
                    print("Before filtering: ", len(dataset))
                    dataset = dataset.filter(lambda x: x["question"] in custom_questions)
                    print("After filtering: ", len(dataset))
                elif args.filter_option == "exclude":
                    custom_questions = joblib.load(args.custom_questions)
                    print("Before filtering: ", len(dataset))
                    dataset = dataset.filter(lambda x: x["question"] not in custom_questions)
                    print("After filtering: ", len(dataset))
                assert len(dataset) == len(custom_questions), "The number of questions and the number of dataset is not matched."
            dataset = make_prompt_for_gpt(dataset, args, q_to_cbrs)
            dataset = make_answer_for_gpt(dataset, args)
            if args.test:
                dataset = dataset.shuffle(seed=42)
                dataset = dataset.select(range(args.test_size))
            preds = []
            for i in tqdm(range(0, len(dataset), args.batch_size), desc="GPT Generating..."):
                batch = dataset[i:i+args.batch_size]
                batch_prompt = batch["prompt"]
                preds.extend(generate_answer_from_gpt(batch_prompt, client, args.max_new_tokens))
        preds = parse_answers(preds, args)
        dataset = dataset.add_column("prediction", preds)
        metrics, overall_metrics = cal_metrics(pd.DataFrame(dataset), args)
        ens = "ens_" if args.ensemble else ""
        dataset_name = "NQ" if "nq" in args.dataset_name.split("/")[1].lower() else "TriviaQA"
        run_name = f"{args.dataset_name.split('/')[1]}_{args.instruction}_{args.custom_questions}_{ens}{args.run_name+'_' if args.run_name else ''}ctx:{args.num_contexts}_cbr:{args.num_cbr_cases}_pert:{args.num_perturb_cases}_{args.cbr_perturb_type}_{len(dataset)}"
        wandb.init(project=f"CBR-RAG-{dataset_name}", name=run_name)
        tbl_result = wandb.Table(dataframe=overall_metrics)
        tbl_prompt = wandb.Table(dataframe=metrics)
        wandb.log({"result":tbl_result, "raw-data":tbl_prompt})
    else:
        dataset = load_dataset(args.dataset_name, split="test")       
        config = PeftConfig.from_pretrained(args.model, revision=MODEL_COMMIT_MAP[args.model])
        inference_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(inference_model, args.model)
        if args.test:
            dataset = dataset.shuffle()
            dataset = dataset.select(range(20))
        dataset = preprocess_dataset(dataset, args)
        result = []
        for data in tqdm(dataset, desc="Generating..."):
            prompt = data["prompt"]
            q, a = data["question"], data["answers"]
            output = tokenizer(prompt, return_tensors="pt").to(args.device)
            output = model.generate(**output, max_new_tokens=args.max_new_tokens)
            pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            pred_wo_prompt = pred[len(prompt):]
            pred_wo_prompt_first_line = pred_wo_prompt.split("\n")[0]
            is_em = exact_match_score(pred_wo_prompt_first_line, a, normalize_answer)
            is_acc = int(text_has_answer(a, pred_wo_prompt_first_line))
            result.append([q,a,prompt, pred,pred_wo_prompt,pred_wo_prompt_first_line, is_em, is_acc])
        df = pd.DataFrame(result, columns=["question", "answer", "prompt", "pred", "pred_wo_prompt", "pred_wo_prompt_first_line", "is_em", "is_acc"])
        print("EM: ", df["is_em"].mean())
        print("ACC: ", df["is_acc"].mean())
        df.to_csv(f"{args.model.split('/')[1]}_{args.dataset_name.split('/')[1]}_result.csv", index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt")
    parser.add_argument("--dataset_name", type=str, required=True, default="")
    parser.add_argument("--num_contexts", type=int, required=False, default=5)
    parser.add_argument("--instruction", type=str, required=False, default="")
    parser.add_argument("--run_name", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, required=False, default=20)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=10)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--cbr", type=str2bool, required=False, default=False)
    parser.add_argument("--q_to_cbrs", type=str, required=False, default="NQ_v3")
    parser.add_argument("--num_cbr_cases", type=int, required=False, default=2)
    parser.add_argument("--num_perturb_cases", type=int, required=False, default=2)
    parser.add_argument("--cbr_perturb", type=str2bool, required=False, default=False)
    parser.add_argument("--cbr_perturb_type", type=str, required=False, default="", choices=["conflict", "missing", "both", "adv"])
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--test_size", type=int, required=False, default=300)
    parser.add_argument("--dataset_split", type=str, required=False, default="train")
    parser.add_argument("--short_context", type=str2bool, required=False, default=False)
    parser.add_argument("--ensemble", type=str2bool, required=False, default=False)
    parser.add_argument("--prepend", type=str2bool, required=False, default=False)
    parser.add_argument("--custom_questions", type=str, required=False, default="")
    parser.add_argument("--explain", type=str2bool, required=False, default=False)
    parser.add_argument("--doc_numbering", type=str2bool, required=False, default=False)
    parser.add_argument("--filter_option", type=str, required=False, default="include", choices=["include", "exclude"])
    parser.add_argument("--random_cbr", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    main(args)
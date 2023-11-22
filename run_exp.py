import argparse
import os
import wandb
import numpy as np
import pandas as pd
import random
import joblib
from evaluation import get_encoder, build_qa_prompt
from src.classes.prompt import TotalPromptSet
from utils import *
from metrics import *
from perturbations import *

def evaluate_single_model(dataset: List[PromptSet],
                          model:dict,
                          model_name: str,
                          args: dict):
    prompts, answers, ori_answers = [], [], []
    idx = 0
    print("length of dataset -> ", len(dataset))
    instruction = get_instruction(args["instruction_type"])
    openai.api_key = "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"
    encoder = get_encoder(args["lm"])
    for idx, data in enumerate(dataset):
        data.supports = data.supports[:args["num_wiki"]]
        prompt = build_qa_prompt(data, model, model_name, args, instruction, encoder)
        if idx == 0 and args["prompt_test"]:
            print(prompt)
        has_answer_in_context = any([wiki.has_answer for wiki in data.supports])
        ori_answers.append(data.answers)
        if args["unanswerable"]:
            if not(has_answer_in_context):
                data.answers = ["unanswerable"]
        answer = data.answers
        prompts.append(prompt)
        answers.append(answer)
    if args["skip_gpt"]:
        return None
    is_corrects, f1_scores, num_tokens, has_answers, total_preds  = [], [], [], [], []
    for i in (tq := tqdm(range(0, len(prompts), args["bs"]), desc=f"EM:  0.0%")):
        b_prompt, b_answer = prompts[i:i+args["bs"]], answers[i:i+args["bs"]]
        preds = get_answer_from_lm(b_prompt, args)
        total_preds.extend(preds)
        is_corrects.extend([exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(preds, b_answer)])
        f1_scores.extend([f1_score(pred, answer, normalize_answer) for pred, answer in zip(preds, b_answer)])
        num_tokens.extend([len(encoder.encode(prompt)) for prompt in b_prompt])
        has_answers.extend([True if answer != ["unanswerable"] else False for answer in b_answer])
        tq.set_description(f"EM : {sum(is_corrects) / len(is_corrects) * 100:4.1f}%")
    raw_data = pd.DataFrame(data={"question":[data.query for data in dataset],
                                  "prompt": prompts,
                                  "answers": [", ".join(ans) if len(ans) > 1 else ans[0] for ans in answers],
                                  "ori_answers": [", ".join(ans) if len(ans) > 1 else ans[0] for ans in ori_answers],
                                  "prediction": total_preds,
                                  "is_exact_match": is_corrects,
                                  "is_accurate": [int(text_has_answer(ans, pred)) for ans, pred in zip(answers, total_preds)],
                                  "ori_is_exact_match": [exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(total_preds, ori_answers)],
                                  "ori_is_accurate": [int(text_has_answer(ans, pred)) for ans, pred in zip(ori_answers, total_preds)],
                                  "num_tokens": num_tokens,
                                  "num_ctxs": [len(data.supports) for data in dataset]
                                  })
    output = pd.DataFrame(data={"em": round(sum(is_corrects) / len(is_corrects) * 100,3),
                                "ori_em": round(sum([exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(total_preds, ori_answers)]) / len(total_preds) * 100, 3),
                                "f1": round(sum(f1_scores) / len(f1_scores) * 100, 3),
                                "accuracy": round(sum([int(text_has_answer(ans, pred)) for ans, pred in zip(answers, total_preds)]) / len(total_preds) * 100, 3),
                                "ori_accuracy": round(sum([int(text_has_answer(ans, pred)) for ans, pred in zip(ori_answers, total_preds)]) / len(total_preds) * 100, 3),
                                "unanswerable_ratio": 100 - round(sum(has_answers) / len(has_answers) * 100, 3),
                                "avg_num_tokens": round(sum(num_tokens) / len(num_tokens), 3),
                                "num_dataset": len(dataset)},
                                index=[0])
    return raw_data, output

def evaluate(dataset: TotalPromptSet, model:dict,  metadata: dict, wandb_run=None):
    raw_data, output = evaluate_single_model(dataset=dataset.prompt_sets,
                                       model=model,
                                       model_name=metadata["model_name"],
                                       args=metadata)
    try:
        raw_data.to_csv("raw_data.csv")
        output.to_csv("output.csv")
    except:
        pass
    tbl_result = wandb.Table(dataframe=output)
    tbl_prompt = wandb.Table(dataframe=raw_data)
    print(raw_data)
    print(output)
    wandb_run.log({"result":tbl_result, "raw-data":tbl_prompt})
    #wandb.log({"result":tbl_result, "raw-data":tbl_prompt})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_wiki",type=int, required=True, help=f"Number of retrieved wiki passages.", dest="num_wiki")
    parser.add_argument("--fewshot_type",type=str, required=False, help="Type of fewshot examples",choices=["cbr", "random"], dest="fewshot_type")
    parser.add_argument("--filter",type=str2bool, required=False, help="1 means fewshots only including entities", dest="filter")
    parser.add_argument("--size",type=int, required=False, default=10, dest="nq_size")
    parser.add_argument("--except_perturb", type=str, required=False, default="", choices=[])
    parser.add_argument("--cache",type=str2bool, required=False, default=True, dest="load_cache")
    parser.add_argument(
        "--ex_type", type=str, required=False, choices=["random_top1", "random_exact", "random_unanswerable", "cbr_top1", "cbr_exact", "cbr_unanswerable", "fixed_top1", "fixed_exact", "fixed_unanswerable"])
    parser.add_argument("--prompt",type=str, required=False, default="base", dest="instruction_type")
    parser.add_argument("--device",type=str, required=False, default="cuda", dest="device")
    parser.add_argument("--num_examples",type=int, required=False, default=5, dest="num_examples")
    parser.add_argument("--unanswerable",type=str2bool, default="true")
    parser.add_argument("--unanswerable_cnt", type=int, default=1)
    parser.add_argument("--mode", type=str, default="compare", choices=["single", "compare"])
    parser.add_argument("--model_name", type=str, choices=["ours","baseline"])
    parser.add_argument(
        "--lm", type=str, default="gpt-3.5-turbo-instruct",
        choices=["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k", "mistral-instruct", "llama2-7b-chat","llama2-13b-chat"])
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--perturb_testset", type=str2bool, default=False)
    parser.add_argument("--perturb_testset_op", type=str, default="random")
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--skip_gpt", type=str2bool, default=False)
    parser.add_argument("--prompt_test", type=str2bool, default=False)
    parser.add_argument("--skip_wandb", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default="datasets/TotalPromptSet.joblib")
    parser.add_argument("--fixed_set", type=str2bool, default=False)
    parser.add_argument("--adaptive_perturbation", type=str2bool, default=False)
    parser.add_argument("--adaptive_perturbation_type", type=str, default="fixed", choices=["fixed", "random"])
    args = parser.parse_args()
    metadata = vars(args)
    total = joblib.load(args.data_path)
    total_promptset, total_metadata = total["promptset"], total["metadata"]
    print("length of total promptset -> ", len(total_promptset.prompt_sets))
    metadata["dataset_config"] = total_metadata
    random.seed(42)
    del total
    if not args.skip_wandb:
        run = wandb.init(
        project="rag",
        notes="experiment",
        tags=["baseline"],
        name=f"{'Test-' if args.test else ''}Exp-{args.nq_size}-{args.mode}{'-'+args.model_name if args.model_name else ''}",
        config=metadata
        )
    model = dict()
    if args.adaptive_perturbation:
        from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer, pipeline
        qa_model = "deepset/roberta-base-squad2"
        nli_model = "cross-encoder/qnli-electra-base"
        model["nlp"] = pipeline("question-answering", model=qa_model, tokenizer=qa_model, device=args.device)
        model["nli"] = {"model": AutoModelForSequenceClassification.from_pretrained(nli_model).to(args.device),
                        "tokenizer": AutoTokenizer.from_pretrained(nli_model)}
    if not args.test:
        evaluate(total_promptset, model, metadata, run)
    else:
        total_promptset.prompt_sets = random.sample(total_promptset.prompt_sets, args.nq_size)
        evaluate(total_promptset, model, metadata, run)

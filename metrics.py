import pandas as pd
from collections import Counter
from typing import Dict, Tuple, List, Callable
from utils import normalize_answer, text_has_answer

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
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])

def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])

# def cal_metrics(single_data: pd.DataFrame, model_name:str) -> dict:
#     SINGLE_METRICS: List[Callable] = [_answerable, _unanswerable]
#     output = dict()
#     for single_metrics in SINGLE_METRICS:
#         output.update({single_metrics.__name__.replace("_", "",1)+"(%)":single_metrics(single_data, model_name)})
#     return list(output.keys()), list(output.values())

def cal_metrics(data: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data["is_exact_match"] = [bool(int(exact_match_score(
        pred, ans, normalize_answer))) for pred, ans in zip(data["prediction"].tolist(), data["new_answers"].tolist()
    )]
    data["is_accurate"] = [text_has_answer(
        ans, pred) for pred, ans in zip(data["prediction"].tolist(), data["new_answers"].tolist()
    )]
    data["f1_score"] = [f1_score(
        pred, ans, normalize_answer) for pred, ans in zip(data["prediction"].tolist(), data["new_answers"].tolist())
    ]
    data["answers"] = data["answers"].apply(lambda x: ", ".join(x) if len(x) > 1 else x[0])
    data["new_answers"] = data["new_answers"].apply(lambda x: ", ".join(x) if len(x) > 1 else x[0])
    em_score, acc_score, f1_score_ = round(data["is_exact_match"].mean()*100,3), round(data["is_accurate"].mean()*100,3), round(data["f1_score"].mean()*100,3)
    unans_only_acc, unans_only_em, adv_unans_acc, adv_unans_em = -1, -1, -1, -1
    if "conflict" in args.dataset_name:
        unanswerable_acc = data[data["new_answers"].apply(lambda x: x == "conflict")]["is_accurate"].mean()*100
        answerable_acc = data[data["new_answers"].apply(lambda x: x != "conflict")]["is_accurate"].mean()*100
        unanswerable_em = data[data["new_answers"].apply(lambda x: x == "conflict")]["is_exact_match"].mean()*100
        answerable_em = data[data["new_answers"].apply(lambda x: x != "conflict")]["is_exact_match"].mean()*100
    elif "missing" in args.dataset_name and "adv" not in args.dataset_name:
        unanswerable_acc = data[data["new_answers"].apply(lambda x: x == "unanswerable")]["is_accurate"].mean()*100
        answerable_acc = data[data["new_answers"].apply(lambda x: x != "unanswerable")]["is_accurate"].mean()*100
        unanswerable_em = data[data["new_answers"].apply(lambda x: x == "unanswerable")]["is_exact_match"].mean()*100
        answerable_em = data[data["new_answers"].apply(lambda x: x != "unanswerable")]["is_exact_match"].mean()*100
    elif "adv" in args.dataset_name:
        unanswerable_acc = data[data["new_answers"].apply(lambda x: x == "unanswerable")]["is_accurate"].mean()*100
        answerable_acc = data[data["new_answers"].apply(lambda x: x != "unanswerable")]["is_accurate"].mean()*100
        unanswerable_em = data[data["new_answers"].apply(lambda x: x == "unanswerable")]["is_exact_match"].mean()*100
        answerable_em = data[data["new_answers"].apply(lambda x: x != "unanswerable")]["is_exact_match"].mean()*100
        subdf_unans = data[data.status.isin(["unans_only"])]
        unans_only_acc = round(subdf_unans["is_accurate"].mean()*100, 3)
        unans_only_em  = round(subdf_unans["is_exact_match"].mean()*100, 3)
        subdf_adv_unans = data[data.status.isin(["adv_unans", "adv_only_unans"])]
        adv_unans_acc = round(subdf_adv_unans["is_accurate"].mean()*100, 3)
        adv_unans_em = round(subdf_adv_unans["is_exact_match"].mean()*100, 3)
    else:
        data["has_answer"] = data["ctxs"].apply(lambda x: any([ex["hasanswer"] for ex in x]))
        unanswerable_acc = data[data["has_answer"] == False]["is_accurate"].mean()*100
        answerable_acc = data[data["has_answer"] == True]["is_accurate"].mean()*100
        unanswerable_em = data[data["has_answer"] == False]["is_exact_match"].mean()*100
        answerable_em = data[data["has_answer"] == True]["is_exact_match"].mean()*100
    unanswerable_acc, answerable_acc = round(unanswerable_acc, 3), round(answerable_acc, 3)
    unanswerable_em, answerable_em = round(unanswerable_em, 3), round(answerable_em, 3)
    data = data.drop(columns=["ctxs"], axis=1)
    output = pd.DataFrame(data={"em": [em_score], "acc": [acc_score], "f1": [f1_score_],
                                "unanswerable_acc": [unanswerable_acc], "answerable_acc": [answerable_acc],
                                "unanswerable_em": [unanswerable_em], "answerable_em": [answerable_em],
                                "unans_only_acc": [unans_only_acc], "unans_only_em": [unans_only_em],
                                "adv_unans_acc": [adv_unans_acc], "adv_unans_em": [adv_unans_em]})
    return data, output    
    

def compare_metrics(compare_data: pd.DataFrame, our_model_name:str, base_model_name: str) -> dict:
    SINGLE_METRICS: List[Callable] = [_answerable, _unanswerable]
    CAMPARE_METRICS: List[Callable] = [_answer_to_unanswerable, _hal_to_unanswerable, _unanswerable_to_hal]
    keys, our_result, base_result = [func_name.__name__.replace("_", "", 1)+"(%)" for func_name in SINGLE_METRICS+CAMPARE_METRICS], [], []
    for single_metrics in SINGLE_METRICS:
        ours, base = single_metrics(compare_data, our_model_name), single_metrics(compare_data, base_model_name)
        our_result.append(ours)
        base_result.append(base)
    for compare_metric in CAMPARE_METRICS:
        metric = compare_metric(compare_data)
        our_result.append(metric)
        base_result.append(metric)
    return keys, our_result, base_result
    
def _answerable(data: pd.DataFrame, model_name: str) -> Tuple[float, float]:
    return round(data[data["answers"] != "unanswerable"][f"{model_name}_is_correct"].mean(), 4) * 100

def _unanswerable(data: pd.DataFrame, model_name: str) -> Tuple[float, float]:
    return round(data[data["answers"] == "unanswerable"][f"{model_name}_is_correct"].mean(), 4) * 100

def _answer_to_unanswerable(compare_data: pd.DataFrame):
    subset = compare_data[(compare_data["answers"] != "unanswerable") & (compare_data["normal_is_correct"] == 1)]
    to_unanswer = len(subset[subset["fewshot_pred"] == "unanswerable"])
    return round(to_unanswer/len(subset), 4) * 100

def _hal_to_unanswerable(compare_data: pd.DataFrame):
    subset = compare_data[(compare_data["answers"] == "unanswerable") & (compare_data["normal_is_correct"] == 0)]
    if len(subset) == 0:
        return 0
    to_unanswer = subset[subset["fewshot_is_correct"] == 1]
    return round(len(to_unanswer)/len(subset), 4)*100

def _unanswerable_to_hal(compare_data: pd.DataFrame):
    subset = compare_data[(compare_data["answers"] == "unanswerable") & (compare_data["normal_is_correct"] == 1)]
    if len(subset) == 0:
        return 0
    to_hal = len(subset[subset["fewshot_pred"] != "unanswerable"])
    return round(to_hal/len(subset), 4)

# def _answerable_to_hal(compare_data: pd.DataFrame):
#     subset = compare_data[(compare_data["answers"] != "unanswerable") & (compare_data["normal_is_correct"] == 0)]
#     to_hal = len(subset[subset["fewshot_pred"] != "unanswerable"])
#     return round(to_hal/len(subset), 4) * 100
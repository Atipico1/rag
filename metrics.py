import pandas as pd
from collections import Counter
from typing import Tuple, List, Callable
from utils import normalize_answer

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

def cal_metrics(single_data: pd.DataFrame, model_name:str) -> dict:
    SINGLE_METRICS: List[Callable] = [_answerable, _unanswerable]
    output = dict()
    for single_metrics in SINGLE_METRICS:
        output.update({single_metrics.__name__.replace("_", "",1)+"(%)":single_metrics(single_data, model_name)})
    return list(output.keys()), list(output.values())

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
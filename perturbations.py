import random
from utils import find_sentence_with_span
from src.substitution_fns import group_answers_by_answer_type, select_random_non_identical_answer
from src.classes.prompt import FewshotPrompt
from src.classes.qadataset import QADataset
from src.classes.qaexample import QAExample
from src.generate_substitutions import corpus_substitution
from nltk import sent_tokenize
from typing import List


def swap_context(data:QAExample, dataset:List[QAExample]) -> FewshotPrompt:
    random_context = None
    ctxs = [ex.context for ex in dataset]
    while not random_context and random_context != data.context:
        random_context = random.choice(ctxs)
    data.context = random_context
    return FewshotPrompt(data, "swap")

def add_adversarial_sentences(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    return None

def shuffle_sentences(data:QAExample) -> FewshotPrompt:
    sentences: list[str] = sent_tokenize(data.context)
    random.shuffle(sentences)
    data.context = " ".join(sentences)
    return FewshotPrompt(data, "shuffle")

def add_conflicting_sentences(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    answers = data.get_answers_in_context()
    if answers == []:
        return FewshotPrompt(data, "origin")
    orig_answer = answers[0]
    _, sent = find_sentence_with_span(orig_answer.spans[0], sent_tokenize(data.context))
    answer_corpus_by_groups = group_answers_by_answer_type(dataset)
    answer_type = data.get_example_answer_type()
    random_answer = select_random_non_identical_answer(data, answer_corpus_by_groups[answer_type]).text
    add_sent = sent.replace(
        data.context[orig_answer.spans[0][0] : orig_answer.spans[0][1]], random_answer)
    data.context += " " + add_sent
    return FewshotPrompt(data, "conflict")

def swap_entities(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    substituted_data = corpus_substitution(data=data,
                                           dset=dataset,
                                           wikidata_info_path="/home/seongilpark/rag/wikidata",
                                           replace_every=True,
                                           num_samples=1,
                                           category="ALL")
    if not substituted_data:
        return FewshotPrompt(data, "origin")
    return FewshotPrompt(substituted_data[0], "entity")

def swap_answer_to_random(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    answers = data.get_answers_in_context()
    if answers == []:
        return FewshotPrompt(data, "origin")
    answer_corpus_by_groups = group_answers_by_answer_type(dataset)
    random_group = random.choice(list(answer_corpus_by_groups.keys()))
    random_answer = select_random_non_identical_answer(data, answer_corpus_by_groups[random_group]).text
    data.update_context_with_random_substitution(random_answer)
    data.gold_answers[0].text = random_answer
    return FewshotPrompt(data, "swap_answer")
    
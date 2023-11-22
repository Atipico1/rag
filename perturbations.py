import random
from utils import check_answer, extract_wiki_page, find_sentence_with_span, find_topk, make_adversarial, merge_sentence, normalize_passage
from src.substitution_fns import group_answers_by_answer_type, select_random_non_identical_answer
from src.classes.prompt import FewshotPrompt
from src.classes.qadataset import QADataset
from src.classes.qaexample import QAExample
from src.generate_substitutions import corpus_substitution
from nltk import sent_tokenize
from typing import List, Dict
import numpy as np
import wikipedia

def swap_context(data:QAExample, dataset:List[QAExample], method:str) -> FewshotPrompt:
    if method == "random":
        random_context = None
        ctxs = [ex.context for ex in dataset]
        while not random_context and random_context != data.context:
            random_context = random.choice(ctxs)
        data.context = random_context
    elif method == "similar":
        data.context = data.metadata.get("most_similar_context")
    return FewshotPrompt(data, "swap_context")

def add_adversarial_sentences_with_same_entity(data:QAExample,
                                               q_tokenizer,
                                               c_tokenizer,
                                               q_model,
                                               c_model,
                                               title_to_wiki: Dict,
                                               strategy: str) -> FewshotPrompt:
    answers = data.get_answers_in_context()
    if answers == []:
        return FewshotPrompt(data, "original")
    answer = answers[0]
    wiki_text = title_to_wiki.get(data.title)
    if not wiki_text:
        return FewshotPrompt(data, "original")
    answer_text = [answer.text.lower() for answer in answers]
    context_sents = sent_tokenize(data.context)
    sent_idx, _ = find_sentence_with_span(answer.spans[0], context_sents)
    query = q_model(q_tokenizer(data.query, return_tensors="pt").to("cuda")["input_ids"]).pooler_output.detach().cpu().numpy()
    sentences = merge_sentence(check_answer(list(map(normalize_passage, sent_tokenize(wiki_text))), answer_text))
    result = []
    if len(sentences) == 1:
        add_sentence = sentences[0]
    else:
        for i in range(0, len(sentences), 32):
            batch = sentences[i:i+32]
            output = c_tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
            embeddings = c_model(**output).pooler_output.detach().cpu().numpy()
            result.extend([emb for emb in embeddings])
        add_sentence = find_topk(query, np.array(result), sentences, topk=1)[0]
    data.context = make_adversarial(sent_idx, context_sents, len(context_sents), add_sentence, strategy)
    return FewshotPrompt(data, "adversarial")

def add_adversarial_sentences_with_similar_entity(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    return None

def shuffle_sentences(data:QAExample) -> FewshotPrompt:
    sentences: list[str] = sent_tokenize(data.context)
    random.shuffle(sentences)
    data.context = " ".join(sentences)
    return FewshotPrompt(data, "shuffle")

def add_conflicting_sentences(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    answers = data.get_answers_in_context()
    if answers == []:
        return FewshotPrompt(data, "original")
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
        return FewshotPrompt(data, "original")
    return FewshotPrompt(substituted_data[0], "swap_entity")

def swap_answer_to_random(data:QAExample, dataset:QADataset) -> FewshotPrompt:
    answers = data.get_answers_in_context()
    if answers == []:
        return FewshotPrompt(data, "original")
    answer_corpus_by_groups = group_answers_by_answer_type(dataset)
    random_group = random.choice(list(answer_corpus_by_groups.keys()))
    random_answer = select_random_non_identical_answer(data, answer_corpus_by_groups[random_group]).text
    data.update_context_with_random_substitution(random_answer)
    data.gold_answers[0].text = random_answer
    return FewshotPrompt(data, "swap_answer")
    
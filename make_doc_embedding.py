import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from nltk import sent_tokenize
import spacy
import joblib
from collections import defaultdict
import numpy as np
import cupy as cp

EDIT_CANDIDATE_POS = ["VERB", "NOUN", "ADJ", "ADV"]
ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB", "NUM"]
POS_ALIGN = ["VERB", "NOUN", "ADJ"]

def find_answer_sent(context, answer):
    sents = sent_tokenize(context)
    for sent in sents:
        if answer in sent:
            return sent
    return None

def parser_answer_chunk(context, answer):
    sents = sent_tokenize(context)
    for idx, sent in enumerate(sents):
        if answer in sent:
            answer_sent_idx = idx
    start_idx = max(0, answer_sent_idx-4)
    end_idx = min(len(sents), answer_sent_idx+3)
    return ' '.join(sents[start_idx:end_idx])

def load_data():
    df = pd.DataFrame(load_dataset("Seongill/SQuAD_unique_questions", split="train"))
    df['answer_sent'] = df.apply(lambda x: find_answer_sent(x['context'], x['answers']['text'][0]), axis=1)
    df['answer'] = df["answers"].apply(lambda x: x["text"][0])
    df = df.dropna(subset=['answer_sent'])
    df['answer_chunk'] = df.apply(lambda x: parser_answer_chunk(x['context'], x['answers']['text'][0]), axis=1)
    return df

if __name__ == "__main__":
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    df = load_data()
    total_ctxs = list(set(df.context.tolist()))
    total_questions = list(set(df.question.tolist()))
    total_answers = df.answer.tolist()
    
    total_texts = total_ctxs + total_questions + total_answers
    print("Length of total text:", len(total_texts))
    BATCH_SIZE = 2000
    docs = []
    for i in tqdm(range(0, len(total_texts), BATCH_SIZE)):
        batch = total_texts[i:i+BATCH_SIZE]
        docs.extend(list(nlp.pipe(batch)))
        
    ent2text, pos2text = defaultdict(list), defaultdict(list)
    text2ent, text2pos = dict(), dict()
    for doc in docs:
        for ent in doc.ents:
            ent2text[ent.label_].append(ent.text)
        for token in doc:
            if not token.ent_type_ and token.pos_ in ANSWER_POS:
                pos2text[token.pos_].append(token.text)
                
    for k, v in ent2text.items():
        ent2text[k] = list(set(v))
    for k, v in pos2text.items():
        pos2text[k] = list(set(v))
        
    for k, v in ent2text.items():
        for vv in v:
            text2ent[vv] = k
    for k, v in pos2text.items():
        for vv in v:
            text2pos[vv] = k

    nlp = spacy.load("en_core_web_lg")

    ent2text_vec = dict()
    for k, v in tqdm(ent2text.items()):
        docs  = nlp.pipe(v)
        ent2text_vec[k] = cp.array([doc.vector / cp.linalg.norm(doc.vector) for doc in docs])
    docs = nlp.pipe(list(text2ent.keys()))
    text2ent_vec = dict()
    for doc in docs:
        text2ent_vec[doc.text] = doc.vector / cp.linalg.norm(doc.vector)
        
    pos2text_vec = dict()
    for k, v in tqdm(pos2text.items()):
        docs  = nlp.pipe(v)
        pos2text_vec[k] = cp.array([doc.vector / cp.linalg.norm(doc.vector) for doc in docs])
    for k, v in pos2text.items():
        for vv in v:
            text2pos[vv] = k
    text2pos_vec = dict()
    docs = nlp.pipe(list(text2pos.keys()))
    for doc in docs:
        text2pos_vec[doc.text] = doc.vector / cp.linalg.norm(doc.vector)
    
    print(len(ent2text), len(text2ent), len(pos2text), len(text2pos))
    print(len(ent2text_vec), len(text2ent_vec), len(pos2text_vec), len(text2pos_vec))
    
    joblib.dump(ent2text_vec, "/data/seongil/datasets/ent2text_vec.joblib")
    joblib.dump(text2ent_vec, "/data/seongil/datasets/text2ent_vec.joblib")
    joblib.dump(pos2text_vec, "/data/seongil/datasets/pos2text_vec.joblib")
    joblib.dump(text2pos_vec, "/data/seongil/datasets/text2pos_vec.joblib")
    joblib.dump(ent2text, "/data/seongil/datasets/ent2text.joblib")
    joblib.dump(text2ent, "/data/seongil/datasets/text2ent.joblib")
    joblib.dump(pos2text, "/data/seongil/datasets/pos2text.joblib")
    joblib.dump(text2pos, "/data/seongil/datasets/text2pos.joblib")    
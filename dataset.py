import json
from tqdm import tqdm
from src.classes.qadataset import QADataset, SquadDataset
from src.classes.qaexample import QAExample

def masking_entity(data: QADataset, model, batch_size) -> QADataset:
    examples = [ex.query for ex in data.examples]
    result = []
    def sub_fn(doc) -> str:
        if not len(doc.ents):
            return doc.text
        
        text_list =[]
        for d in doc:
            if d.pos_ == "PUNCT":
                text_list.append("@"+d.text)
            elif d.pos_ == "AUX" and d.text == "'s":
                text_list.append("@"+d.text)
            else:
                text_list.append(d.text)

        for ent in doc.ents:
            text_list[ent.start:ent.end] = ["[B]"]* (ent.end - ent.start)
            text_list[ent.start] = "[BLANK]"
        return " ".join(text_list).replace(" [B]", "").replace(" @", "")
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]
        batch_docs = list(model.pipe(batch, batch_size=batch_size))
        masked_quries = [sub_fn(doc) for doc in batch_docs]
        result.extend(masked_quries)
    assert len(result) == len(examples), "Length doesn't match"
    for i in range(len(result)):
        data.examples[i].masked_query = result[i]
    return data

def remove_duplicate_queries(data: SquadDataset) -> SquadDataset:
    masked_queries: list[str] = [ex.masked_query for ex in data.examples]
    buffer = dict()
    result_idxs = []
    for idx, query in enumerate(masked_queries):
        if not buffer.get(query):
            buffer[query] = True
            result_idxs.append(idx)
    new_examples = []
    for idx, ex in enumerate(data.examples):
        if idx in result_idxs:
            new_examples.append(ex)
    data.examples = new_examples
    return data

def sent_embedding(data: QADataset, model, batch_size):
    examples = data.examples
    result = []
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]
        sents = [b.masked_query for b in batch]
        embeddings = model.encode(sents)
        result.extend(embeddings)
    assert len(result) == len(examples), "Length doesn't match"
    for i in range(len(result)):
        data.examples[i].embedding = result[i]
    return data
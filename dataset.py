import json
from spacy.tokens import Doc, Span

def masking_entity(doc) -> str:
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
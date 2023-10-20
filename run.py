import argparse
from dataset import masking_entity, remove_duplicate_queries, sent_embedding
from src.classes.prompt import PromptSet, TotalPromptSet, WikiContext
from src.classes.qadataset import QADataset, SquadDataset
from src.classes.qaexample import QAExample
from utils import evaluation, find_topk
from sentence_transformers import SentenceTransformer
from perturbations import *
import nltk
import numpy as np
import spacy
from tqdm import tqdm
from copy import deepcopy
import wandb

def load_squad(nlp, emb_model, batch_size: int, filtering: int):
    # dataset = QADataset.load("SquadDataset")
    # dataset.examples = dataset.examples
    # dataset = masking_entity(dataset, nlp, batch_size=batch_size)
    # dataset = remove_duplicate_queries(dataset)
    # dataset = sent_embedding(dataset, emb_model, batch_size=batch_size)
    # dataset.custom_save()
    dataset = QADataset.custom_load("SquadDataset")
    if filtering:
        exs = []
        print("before:", len(dataset.examples))
        for ex in dataset.examples:
            if ex.get_example_answer_type():
                exs.append(ex)
        print("after:", len(exs))        
        dataset.examples = exs
    return dataset

def load_nq(nlp, emb_model, batch_size: int):
    dataset = QADataset.load("NQ")
    dataset.examples = dataset.examples[::100]
    dataset = masking_entity(dataset, nlp, batch_size=batch_size)
    dataset = sent_embedding(dataset, emb_model, batch_size=batch_size)
    #dataset.custom_save()
    #dataset= QADataset.custom_load("NQ")
    return dataset

def generate_promptset(squad: List[QAExample],
                       squad_dataset: QADataset,
                       nq_ex: QAExample,
                       topk_indics: List[int],
                       wiki_topk: int):
    topk_examples = [squad[idx] for idx in topk_indics] if len(topk_indics) > 0 else []
    fewshots = []
    for ex in topk_examples:
        if ex.get_example_answer_type():
            fewshots.append(swap_entities(deepcopy(ex), squad_dataset))
            #fewshots.append(add_adversarial_sentences(deepcopy(ex), squad))
            fewshots.append(add_conflicting_sentences(deepcopy(ex), squad_dataset))
        else:
            fewshots.append(swap_answer_to_random(deepcopy(ex), squad_dataset))
        fewshots.append(swap_context(deepcopy(ex), squad))
        fewshots.append(shuffle_sentences(deepcopy(ex)))
     
    wikis = nq_ex.context[:wiki_topk]
    wikis = [WikiContext(id=wiki["id"], title=wiki["title"], text=wiki["text"], score=wiki["score"], has_answer=wiki["has_answer"]) for wiki in wikis]
    return PromptSet(query=nq_ex.query,
                     answers=[ans.text for ans in nq_ex.gold_answers],
                     fewshots=fewshots,
                     wikis=wikis)

def generate(squad: SquadDataset, nq: QADataset, fewshot_k: int, wiki_k: int) -> TotalPromptSet:
    nq_examples, squad_examples = nq.examples, squad.examples
    squad_embeddings = [ex.embedding for ex in squad_examples]
    result = []
    for nq_ex in tqdm(nq_examples, desc="NQ processing..."):
        topk_indices = find_topk(nq_ex.embedding, squad_embeddings, topk=fewshot_k)
        prompt = generate_promptset(squad_examples, squad, nq_ex, topk_indices, wiki_topk=wiki_k)
        result.append(prompt)
    return TotalPromptSet(result)

def evaluate(dataset: TotalPromptSet, metadata: dict):
    evaluation(dataset=dataset.prompt_sets,
               metadata=metadata,
               num_docs=metadata["num_wiki"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--num_wiki",
        type=int,
        required=True,
        help=f"Number of retrieved wiki passages.",
        dest="num_wiki"
    )
    parser.add_argument(
        "-few",
        "--fewshot",
        type=int,
        required=True,
        help="Number of Fewshot Examples.",
        dest="num_fewshot"
    )
    parser.add_argument(
        "-ftype",
        "--fewshottype",
        type=str,
        required=False,
        help="Type of fewshot examples",
        choices=["cbr", "random"],
        dest="fewshot_type"
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=int,
        required=False,
        help="1 means fewshots only including entities",
        dest="filter"
    )

    args = parser.parse_args()
    metadata = vars(args)
    run = wandb.init(
    project="rag",
    notes="experiment",
    tags=["baseline"],
    config=metadata
    )
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    squad_dataset = load_squad(nlp, model, batch_size=1000, filtering=args.filter)
    nq_dataset = load_nq(nlp, model, batch_size=1000)
    total_promptset = generate(squad=squad_dataset,
                               nq=nq_dataset,
                               fewshot_k=args.num_fewshot,
                               wiki_k=args.num_wiki)
    evaluate(total_promptset, metadata=metadata)
    total_promptset.save_sample()

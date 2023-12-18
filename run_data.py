
import argparse, spacy, random, joblib, torch
from copy import deepcopy
from typing import Dict, List
from tqdm import tqdm
from perturbations import add_adversarial_sentences_with_same_entity, add_conflicting_sentences, swap_context, swap_entities
from dataset import context_embedding, dataset_filter, find_most_similar_context, load_nq_train_data, masking_entity, remove_duplicate_queries, sent_embedding
from src.substitution_fns import group_answers_by_answer_type, select_random_non_identical_answer
from src.classes.cbr_data import NQ, NQExample, WikiContext
from src.classes.prompt import FewshotPrompt, PromptSet, TotalPromptSet
from src.classes.qaexample import QAExample
from utils import check_same_answers_in, find_topk, get_answer, str2bool
from src.classes.qadataset import QADataset, SquadDataset
from transformers import AutoModel, AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

def load_nq(nlp, emb_model, tokenizer, args):
    random.seed(42)
    if not args.nq_cache:
        if args.split == "test":
            dataset = QADataset.load("NQ")
        else:
            dataset = QADataset.load("NQTrain")
        dataset = masking_entity(dataset, nlp, args.masking_model, batch_size=args.bs)
        dataset = sent_embedding(dataset, args.emb_model, emb_model, tokenizer, batch_size=args.bs)
        dataset.custom_save()
    else:
        if args.split == "test":
            dataset = QADataset.custom_load("NQ")
        else:
            dataset = QADataset.custom_load("NQTrain")
    if args.ent_swap:
        train_set = QADataset.load("NQTrain")
        answer_corpus_by_groups = group_answers_by_answer_type(train_set)
        cnt = 0
        for ex in dataset.examples:
            ans_type = ex.get_example_answer_type()
            if ans_type:
                cnt += 1
                ex.metadata = {"substitution":select_random_non_identical_answer(ex, answer_corpus_by_groups[ans_type]).text}
            else:
                ex.metadata = {"substitution":None}
        print("NQ with Entity Size :", cnt)
    print("NQ Total Dataset Size :", len(dataset.examples))
    return dataset

def load_squad(nlp, ctx_tokenizer,ctx_emb_model, emb_model, tokenizer, args):
    context_embeddings = None
    if not args.squad_cache:
        dataset = QADataset.load("SquadDataset")
        if args.swap_context_method == "similar":
            context_embeddings = context_embedding(dataset, ctx_emb_model, ctx_tokenizer, batch_size=args.bs)
        if args.squad_filter:
            title_to_wiki = joblib.load("./datasets/title_to_wiki")
            exs = []
            for ex in dataset.examples:
                if ex.get_example_answer_type() and title_to_wiki.get(ex.title):
                    exs.append(ex)
            dataset.examples = exs
        dataset = masking_entity(dataset, nlp, args.masking_model, batch_size=args.bs)
        dataset = remove_duplicate_queries(dataset)
        dataset = sent_embedding(dataset, args.emb_model, emb_model, tokenizer, batch_size=args.bs)
        if args.swap_context_method == "similar":
            find_most_similar_context(dataset, context_embeddings)
        dataset.custom_save()
    else:
        dataset = QADataset.custom_load("SquadDataset")
        if args.squad_filter:
            title_to_wiki = joblib.load("./datasets/title_to_wiki")
            exs = []
            for ex in dataset.examples:
                if ex.get_example_answer_type() and title_to_wiki.get(ex.title):
                    exs.append(ex)
            dataset.examples = exs
        if args.swap_context_method == "similar":
            context_embeddings = context_embedding(dataset, ctx_emb_model, ctx_tokenizer, batch_size=args.bs)
            find_most_similar_context(dataset, context_embeddings)
    print("SQuAD Size :", len(dataset.examples))
    return dataset, context_embeddings

# has_answer가 하나도 없는 경우에는 cbr 예시에서 제외
def generate_cbr_prompt(squad: SquadDataset, dataset: QADataset, args):
    examples, squad_examples = dataset.examples, squad.examples
    result = []
    for ex in tqdm(examples, desc="Generate CBR Prompt"):
        cbrs = []
        filtered_squad, squad_cnt = dataset_filter(ex, squad_examples, args)
        filtered_squad_embeddings = [ex.embedding for ex in filtered_squad]
        cbr_examples = find_topk(ex.embedding, filtered_squad_embeddings, filtered_squad, topk=args.num_ex, filter_same_questions=args.filter_same_question)
        

def generate_promptset(squad: SquadDataset, squad_ctx_embeddings: Dict, nq: QADataset, args, nlp, q_tokenizer, c_tokenizer, q_model, c_model) -> TotalPromptSet:
    nq_exs, squad_examples = nq.examples, squad.examples
    title_to_wiki = joblib.load("./datasets/title_to_wiki")
    #TODO : nq_train 불러오는 로직을 기존 함수와 통합시켜야 함
    nq_train: List[NQExample] = load_nq_train_data(args.bs, nlp, q_model, q_tokenizer).dataset
    result = []
    total_nq_cnt, total_squad_cnt = 0, 0
    for nq_ex in tqdm(nq_exs, desc="Generate Promptset"):
        fewshots = []
        filtered_nq_train, nq_cnt = dataset_filter(nq_ex, nq_train, args)
        filtered_squad, squad_cnt = dataset_filter(nq_ex, squad_examples, args)
        squad_embeddings = [ex.embedding for ex in filtered_squad]
        cbr_embeddings = [ex.masked_embedding for ex in filtered_nq_train]
        topk_fewshots = find_topk(nq_ex.embedding, squad_embeddings, filtered_squad, topk=args.num_fewshot, filter_same_questions=args.filter_same_question, random_selection=False)
        cbr_examples = find_topk(nq_ex.embedding, cbr_embeddings, filtered_nq_train, topk=args.num_ex)
        random_examples = random.sample(nq_train, args.num_ex)
        for ex in topk_fewshots:
            fewshot = []
            fewshot.append(swap_entities(deepcopy(ex), squad))
            fewshot.append(add_adversarial_sentences_with_same_entity(deepcopy(ex), q_tokenizer, c_tokenizer, q_model, c_model, title_to_wiki, args.adversary_strategy))
            fewshot.append(add_conflicting_sentences(deepcopy(ex), squad))
            fewshot.append(swap_context(deepcopy(ex), squad_examples, args.swap_context_method))
            fewshot.append(FewshotPrompt(deepcopy(ex), "original"))
            fewshots.append(fewshot)
        wikis = nq_ex.context[:args.num_context]
        wikis = [WikiContext(id=wiki["id"], title=wiki["title"], text=wiki["text"], score=wiki["score"], has_answer=wiki["has_answer"]) for wiki in wikis]
        total_nq_cnt += nq_cnt
        total_squad_cnt += squad_cnt
        result.append(PromptSet(query=nq_ex.query,
                                answers=[ans.text for ans in nq_ex.gold_answers],
                                cbr_examples=cbr_examples,
                                random_examples=random_examples,
                                fewshots=fewshots,
                                wikis=wikis,
                                substitution=nq_ex.metadata.get("substitution"),
                                measure={"nq_same_answer_examples":nq_cnt, "squad_same_answer_examples":squad_cnt} if args.filter_same_answer else None))
    total_output = dict()
    output = TotalPromptSet(result)
    total_output["promptset"] = output
    total_output["metadata"] = vars(args)
    joblib.dump(total_output, f"datasets/TotalPromptSet{'-'+args.custom_name if args.custom_name else ''}.joblib")
    print("Average NQ Same Answer Examples :", round(total_nq_cnt/len(nq_exs),3))
    print("Average SQuAD Same Answer Examples :", round(total_squad_cnt/len(nq_exs),3))
    return output

def load_masking_model(model_type: str):
    if model_type == "spacy":
        return spacy.load("en_core_web_trf")
    elif model_type == "tner":
        from tner import TransformersNER
        return TransformersNER('tner/roberta-large-ontonotes5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_cache", type=str2bool, required=True, default=True)
    parser.add_argument("--nq_cache", type=str2bool, required=True, default=True)
    parser.add_argument("--num_context", type=int, required=True, default=10)
    parser.add_argument("--pert_type", type=str, required=False, choices=["random", "one_to_many"])
    parser.add_argument("--num_ex", type=int, required=False, default=5)
    parser.add_argument("--num_fewshot", type=int, required=False, default=1)
    parser.add_argument("--unanswerable", type=str2bool, required=False, default=True)
    parser.add_argument("--adversary",type=str, default="add", dest="adversary_strategy")
    parser.add_argument("--bs", type=int, required=False, default=32)
    parser.add_argument("--squad_filter",type=str2bool, required=False, default=True)
    parser.add_argument("--nq_filter",type=str2bool, required=False, default=True)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--nq_size", type=int, required=False, default=1800)
    parser.add_argument("--masking_model", type=str, required=False, default="tner", choices=["spacy", "tner"])
    parser.add_argument("--emb_model", type=str, required=False, default="dpr", choices=["roberta", "dpr"])
    parser.add_argument("--custom_name", type=str, required=False, default=None)
    parser.add_argument("--filter_wh", type=str2bool, required=False, default=True)
    parser.add_argument("--filter_same_answer", type=str2bool, required=False, default=True)
    parser.add_argument("--filter_same_question", type=str2bool, required=False, default=True)
    parser.add_argument("--ent_swap", type=str2bool, required=False, default=False)
    parser.add_argument("--swap_context_method", type=str, required=False, default="random", choices=["random", "similar"])
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--split", type=str, required=False, default="test", choices=["train", "test"])
    args = parser.parse_args()
    metadata = vars(args)
    spacy.prefer_gpu()
    nlp = load_masking_model(args.masking_model)
    c_tok = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    c_enc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    q_tok = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    c_enc.to(args.device)
    q_enc.to(args.device)
    with torch.no_grad():
        q_enc.eval()
        c_enc.eval()
        squad_dataset, squad_ctx_embeddings = load_squad(nlp, c_tok, c_enc, q_enc, q_tok, args)
        nq_dataset = load_nq(nlp, q_enc, q_tok, args)
        if args.test:
            print("SQUAD EXAMPLE")
            squad_dataset.examples[0].test_print()
            print("-"*100)
            print("NQ EXAMPLE")
            nq_dataset.examples[0].test_print()
        promptset = generate_promptset(squad_dataset, squad_ctx_embeddings, nq_dataset, args, nlp, q_tok, c_tok, q_enc, c_enc)
    promptset.save_sample(args, n=50)
from typing import List, Dict
from src.classes.cbr_data import NQExample, WikiContext
from src.classes.qaexample import QAExample
import os 
import json
import tiktoken
import wandb
import pandas as pd
from dataclasses import dataclass
encoder = tiktoken.get_encoding("cl100k_base")

class SquadExample():
    def __init__(self, ex: QAExample):
        self.question = ex.query
        self.context = str(ex.context)
        self.answer = ex.gold_answers[0].text

class FewshotPrompt(SquadExample):
    def __init__(self, ex: QAExample, perturbation_type, original: SquadExample = None):
        self.question = ex.query
        self.perturbation_type = perturbation_type
        self.original = original
        if perturbation_type in ["swap_entity","swap_answer","original", "adversarial"]:
            self.answer = ex.gold_answers[0].text
            self.context = str(ex.context)
        elif perturbation_type in ["swap_context", "conflict"]:
            self.answer = "unanswerable"
            self.context = str(ex.context)
        else:
            self.answer = ex.original_example.gold_answers[0].text
            self.context = str(ex.context)

class FewshotSet():
    def __init__(self, fewshot_dict: Dict):
        self.id = fewshot_dict["id"]
        self.embedding = fewshot_dict["embedding"]
        self.masked_query = fewshot_dict["masked_query"]
        self.origin = fewshot_dict["original"]
        self.adversarial = fewshot_dict["adversarial"]
        self.entity = fewshot_dict["swap_entity"]
        self.conflict = fewshot_dict["conflict"]
        self.swap = fewshot_dict["swap_context"]
        self.query = self.origin.question

class PromptSet():
    def __init__(self,
                 query: str,
                 answers: List[str],
                 cbr_examples:List[NQExample],
                 random_examples:List[NQExample],
                 fewshots:List[List[FewshotPrompt]],
                 wikis: List[WikiContext],
                 substitution: str,
                 measure: Dict):
        self.query = query
        self.answers = answers
        self.fewshots = fewshots
        self.random_examples = random_examples
        self.cbr_examples = cbr_examples
        self.supports = wikis
        self.prompt = self.make_fewshot_prompt()
        self.num_tokens = self.cal_num_tokens(self.prompt)
        self.num_docs = len(wikis)
        self.measure = measure
        self.substitution = substitution

    def make_fewshot_prompt(self):
        result = ""
        for fewshots in self.fewshots:
            for fewshot in fewshots:
                q, c, a = fewshot.question, fewshot.context, fewshot.answer
                result += (f"Question+{fewshot.perturbation_type}: " + q + "\n")
                result += ("Knowledge: " + c + "\n")
                result += ("Answer: " + a + "\n")
                result += "\n"
        result += ("Question: " + self.query + "\n")
        result += ("Knowledge: " + "\n".join([ctx.text for ctx in self.supports])+ "\n")
        result += ("Answer: ")
        return result

    def cal_num_tokens(self, prompt):
        return len(encoder.encode(prompt))

def make_ex_prompt(datas: List[NQExample]) -> str:
    result = ""
    for data in datas:
        result += ("Question: "+ data.question + "\n")
        result += ("Answer: "+ data.answers[0] + "\n\n")
    return result

class TotalPromptSet():
    def __init__(self, prompts: List[PromptSet]):
        self.prompt_sets = prompts
    
    def save_sample(self, args, n: int =10):
        wandb.init(
        project="rag",
        notes="experiment",
        name="prompt_examples",
        config=vars(args)
        )
        result = []
        for prompt in self.prompt_sets[:n]:
            query, answer, fewshots = prompt.query, prompt.answers, prompt.fewshots
            cbr_ex, ran_ex, contexts = prompt.cbr_examples, prompt.random_examples, prompt.supports
            result.append([query, ", ".join(answer), prompt.make_fewshot_prompt(),
                           make_ex_prompt(cbr_ex), make_ex_prompt(ran_ex)])
        df = pd.DataFrame(data=result, columns=["question","answer","fewshot","cbr","random"])
        tbl_prompt = wandb.Table(dataframe=df)
        wandb.log({"prompt_examples":tbl_prompt})
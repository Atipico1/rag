from typing import List
from src.classes.qaexample import QAExample
import os 
import json
import tiktoken

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
        if perturbation_type in ["shuffle", "adversarial", "entity", "origin", "swap_answer"]:
            self.answer = ex.gold_answers[0].text
            self.context = str(ex.context)
        elif perturbation_type in ["swap", "conflict"]:
            self.answer = "unanswerable"
            self.context = str(ex.context)
        else:
            self.answer = ex.original_example.gold_answers[0].text
            self.context = str(ex.context)

class WikiContext():
    def __init__(self, id, title, text, score, has_answer):
        self.id = id
        self.title = title
        self.text = text
        self.score = score
        self.has_asnwer = has_answer

class PromptSet():
    def __init__(self, query, answers, fewshots:List[FewshotPrompt], wikis: List[WikiContext]):
        self.query = query
        self.answers = answers
        self.fewshots = fewshots
        self.supports = wikis
        self.prompt = self.make_prompt()
        self.num_tokens = self.cal_num_tokens(self.prompt)
        self.num_docs = len(wikis)

    def make_prompt(self):
        result = ""
        for fewshot in self.fewshots:
            q, c, a = fewshot.question, fewshot.context, fewshot.answer
            result += (f"Question+{fewshot.perturbation_type}:" + q + "\n")
            result += ("Knowledge:" + c + "\n")
            result += ("Answer:" + a + "\n")
            result += "\n"
        result += ("Question:" + self.query + "\n")
        result += ("Knowledge:" + "\n".join([ctx.text for ctx in self.supports])+ "\n")
        result += ("Answer:")
        return result

    def cal_num_tokens(self, prompt):
        return len(encoder.encode(prompt))
    
class TotalPromptSet():
    def __init__(self, prompts: List[PromptSet]):
        self.prompt_sets = prompts
    
    def save_sample(self, path="sample", n: int =10):
        os.makedirs(path, exist_ok=True)
        result = []
        for i in range(n):
            promptset = self.prompt_sets[i]
            result.append(promptset.prompt)
        with open(f"{path}/sample.json", "w") as f:
            json.dump(result, f)

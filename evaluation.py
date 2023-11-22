from dataset import knowledge_separation
from src.classes.prompt import FewshotPrompt
from utils import *
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from metrics import *
from transformers import AutoTokenizer

def make_examples(data: PromptSet, ex_num: int, ex_type:str, unanswerable_cnt: int=1) -> str:
    if ex_num == 0:
        return ""
    if len(ex_type.split("_")) == 1:
        print("EX TYPE:", ex_type, "---\n")
    head, tail = ex_type.split("_")
    output = ""
    if head == "random":
        examples = data.random_examples[:ex_num]
        if tail == "top1":
            for ex in examples:
                output += ("Knowledge: " + ex.contexts[0].text + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n")
        elif tail == "exact":
                context = sorted(list(filter(lambda x: x.has_answer, ex.contexts)), key=lambda x: x.score, reverse=True)[0].text
                output += ("Knowledge: " + context + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n")
        else:
            no_answer_idx = []
            for i, ex in enumerate(examples[:ex_num]):
                if len(no_answer_idx) == unanswerable_cnt:
                    break
                context = sorted(list(filter(lambda x: not(x.has_answer), ex.contexts)), key=lambda x: x.score, reverse=True)
                if len(context) > 0:
                    context_text = context[0].text
                    output += ("Knowledge: " + context_text + "\n")
                    output += ("Question: " + ex.question + "\n")
                    output += ("Answer: unanswerable\n\n") 
                    no_answer_idx.append(i)
                else:
                    continue
            assert len(no_answer_idx) == unanswerable_cnt, "Too many unanswerable"   
            for j, ex in enumerate(examples[:ex_num]):
                if j in no_answer_idx:
                    continue
                context = sorted(list(filter(lambda x: x.has_answer, ex.contexts)), key=lambda x: x.score, reverse=True)[0].text
                output += ("Knowledge: " + context + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n")
    elif head == "cbr":
        examples = data.cbr_examples[:ex_num]
        if tail == "top1":
            for ex in examples:
                output += ("Knowledge: " + ex.contexts[0].text + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n")
        elif tail == "exact":
            for ex in examples:
                context = sorted(list(filter(lambda x: x.has_answer, ex.contexts)), key=lambda x: x.score, reverse=True)[0].text
                output += ("Knowledge: " + context + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n") 
        else:
            no_answer_idx = []
            for i, ex in enumerate(examples[:ex_num]):
                if len(no_answer_idx) == unanswerable_cnt:
                    break
                context = sorted(list(filter(lambda x: not(x.has_answer), ex.contexts)), key=lambda x: x.score, reverse=True)
                if len(context) > 0:
                    context_text = context[0].text
                    output += ("Knowledge: " + context_text + "\n")
                    output += ("Question: " + ex.question + "\n")
                    output += ("Answer: unanswerable\n\n") 
                    no_answer_idx.append(i)
                else:
                    continue
            assert len(no_answer_idx) == unanswerable_cnt, "Too many unanswerable"   
            for j, ex in enumerate(examples[:ex_num]):
                if j in no_answer_idx:
                    continue
                context = sorted(list(filter(lambda x: x.has_answer, ex.contexts)), key=lambda x: x.score, reverse=True)[0].text
                output += ("Knowledge: " + context + "\n")
                output += ("Question: " + ex.question + "\n")
                output += ("Answer: " + ex.answers[0] + "\n\n")
    else:
        output = get_format_prompt(ex_num, tail)
    return output

def make_adaptive_perturbations(data: PromptSet, adaptive_type: str, args: dict) -> List[FewshotPrompt]:
    # ['no_relevant_ctx', 'one_relevant_ctx', 'conflict', 'many_relevant_ctx']
    # no_relevant_ctx : 1 original + 1 context swap + 1 swap_answer + 1 adversarial + 1 conflict
    # one_relenvat_ctx : 3 original + 1 swap_answer + 1 adversarial
    # conflict : 1 original + 1 swap_answer + 1 adversarial + 2 conflict
    # many_relevant_ctx : 4 original+ 1 swap_answer
    fewshots : List[List[FewshotPrompt]] = data.fewshots
    group_by_perturbation_type = defaultdict(list)
    for fewshot in fewshots:
        for shot in fewshot:
            group_by_perturbation_type[shot.perturbation_type].append(shot)
    strategy = args["adaptive_perturbation_type"]
    num = args["num_examples"]
    output = []
    if adaptive_type == "no_relevant_ctx":
        if strategy == "fixed":
            output = fewshots[-1]
        elif strategy == "random":
            for _, v in group_by_perturbation_type.items():
                output.extend(random.sample(v, 1))
    elif adaptive_type =="one_relevant_ctx":
        if strategy == "fixed":
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original", "swap_answer", "adversarial"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original"]])
        elif strategy == "random":
            output.extend(random.sample(group_by_perturbation_type["original"], 3))
            output.extend(random.sample(group_by_perturbation_type["swap_answer"], 1))
            output.extend(random.sample(group_by_perturbation_type["adversarial"], 1))
    elif adaptive_type == "conflict":
        if strategy == "fixed":
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original", "swap_answer", "adversarial", "conflict"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in [ "conflict"]])        
        elif strategy == "random":
            output.extend(random.sample(group_by_perturbation_type["original"], 1))
            output.extend(random.sample(group_by_perturbation_type["swap_answer"], 1))
            output.extend(random.sample(group_by_perturbation_type["adversarial"], 1))
            output.extend(random.sample(group_by_perturbation_type["conflict"], 2))
    elif adaptive_type == "many_relevant_ctx":
        if strategy == "fixed":
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original", "swap_answer"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original"]])
            output.extend([fewshot for fewshot in fewshots.pop() if fewshot.perturbation_type in ["original"]])
        elif strategy == "random":
            output.extend(random.sample(group_by_perturbation_type["original"], 4))
            output.extend(random.sample(group_by_perturbation_type["swap_answer"], 1))   
    else:
        raise NotImplementedError
    return output

def build_qa_prompt_for_gpt(data: PromptSet, model:dict, model_name: str, args: dict, instruction:str, encoder) -> str:
    q = normalize_question(data.query)
    prompt = instruction
    format_prompts = make_examples(data, args["num_examples"], args["ex_type"], args["unanswerable_cnt"])
    if data.supports == []:
        return f"Answer the question:\n\nQuestion: {q}\nAnswer:"
    knowledge = "\n".join([wiki.text for wiki in data.supports])
    output = None
    if args["adaptive_perturbation"]:
        if len(data.fewshots) < 4:
            print("Fewshot size is less than 4, so we don't use adaptive perturbation")
            data.fewshots = data.fewshots[-1]
        elif isinstance(data.fewshots[0], FewshotPrompt):
            print("Invalid Fewshot type, so we don't use adaptive perturbation")
            print(f"Query : {data.query}")
            raise NameError
        else:
            output = determine_perturbation_type(data, model, args)
            data.fewshots = make_adaptive_perturbations(data, output, args)
    if model_name == "ours":
        for shot in data.fewshots:
            shot_q, shot_a, shot_c = shot.question, shot.answer, shot.context
            shot_c = knowledge_separation(shot.perturbation_type, shot_c)
            prompt += f"Knolwedge: {shot_c}\nQuestion: {shot_q}\nAnswer: {shot_a}\n\n"            
        prompt += f"Knolwedge: {knowledge}\nQuestion: {q}\nAnswer:"
    else:
        prompt += format_prompts
        prompt += f"Knowledge: {knowledge}\nQuestion: {q}\nAnswer:"
    if len(encoder.encode(prompt)) > (4096-args["max_tokens"]):
        data.supports.pop()
        return build_qa_prompt_for_gpt(data, model, model_name, args, instruction, encoder)
    return prompt

def build_qa_prompt_for_llama(data: PromptSet, model:dict, model_name: str, args: dict, instruction:str, encoder) -> str:
    q = normalize_question(data.query)
    knowledge = "\n".join([wiki.text for wiki in data.supports])
    prompt = "[INST] <<SYS>>\n" + instruction +"<</SYS>>\n"
    if model_name == "ours":
        for shot in data.fewshots:
            shot_q, shot_a, shot_c = shot.question, shot.answer, shot.context
            shot_c = knowledge_separation(shot.perturbation_type, shot_c)
            prompt += f"Knolwedge: {shot_c}\nQuestion: {shot_q}\nAnswer: {shot_a}\n\n"            
        prompt += f"Knolwedge: {knowledge}\nQuestion: {q}\nAnswer: [/INST]"
    else:
        prompt += make_examples(data, args["ex_type"], args["unanswerable_cnt"])
        prompt += f"Knowledge: {knowledge}\nQuestion: {q}\nAnswer: [/INST]"
    if len(encoder.encode(prompt)) > 4000:
        data.supports.pop()
        return build_qa_prompt_for_llama(data, model, model_name, args, instruction, encoder)
    return prompt

def build_qa_prompt_for_mistral(data: PromptSet, model:dict, model_name: str, args: dict, instruction:str, encoder) -> str:
    q = normalize_question(data.query)
    knowledge = "\n".join([wiki.text for wiki in data.supports])
    prompt = "<s>[INST] " + instruction
    if model_name == "ours":
        for shot in data.fewshots:
            shot_q, shot_a, shot_c = shot.question, shot.answer, shot.context
            shot_c = knowledge_separation(shot.perturbation_type, shot_c)
            prompt += f"Knolwedge: {shot_c}\nQuestion: {shot_q}\nAnswer: {shot_a}\n\n"            
        prompt += f"Knolwedge: {knowledge}\nQuestion: {q}\nAnswer: [/INST]"
    else:
        prompt += make_examples(data, args["ex_type"], args["unanswerable_cnt"])
        prompt += f"Knowledge: {knowledge}\nQuestion: {q}\nAnswer: [/INST]"
    if len(encoder.encode(prompt)) > 8000:
        data.supports.pop()
        return build_qa_prompt_for_llama(data, model, model_name, args, instruction, encoder)    
    return prompt

def build_qa_prompt(data, model, model_name, args, instruction, encoder) -> str:
    if args["lm"] in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k"]:
        prompt = build_qa_prompt_for_gpt(data, model, model_name, args, instruction, encoder)
    elif args["lm"] in ["mistral-instruct"]:
        prompt = build_qa_prompt_for_mistral(data, model, model_name, args, instruction, encoder)
    elif args["lm"] in ["llama2-7b-chat", "llama2-13b-chat"]:
        prompt = build_qa_prompt_for_llama(data, model, model_name, args, instruction, encoder)
    else:
        raise NotImplementedError
    return prompt

def get_encoder(lm):
    if lm in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k"]:
        return tiktoken.get_encoding("cl100k_base")
    elif lm in ["mistral-instruct"]:
        return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    elif lm in ["llama2-7b-chat", "llama2-13b-chat"]:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


      
from src.classes.train_arguments import ScriptArguments
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, is_xpu_available
from tqdm import tqdm
import wandb 

def preprocess_baseline(dataset, script_args):
    def make_prompt(example):
        contexts = "\n".join([ex["text"] for ex in example["ctxs"][:script_args.num_contexts]])
        prompt = "Background:" + contexts + "\n\n"
        prompt += f"Q: {example['question']}\n"
        if script_args.dataset_split == "test":
            prompt += "A: "
        else:
            prompt += f"A: {example['answers'][0]}"
        return {"prompt":prompt}
    return dataset.map(make_prompt)

def preprocess_cbr(dataset, script_args):
    def make_prompt(example):
        contexts = "\n".join([ex["text"] for ex in example["ctxs"][:script_args.num_contexts]])
        prompt = f"[CASE]\nBackground: {example['case_context']}\n\nQ: {example['case_question']}\nA: {example['case_answer']}\n[/CASE]\n"
        prompt += "Background: " + contexts + "\n\n"
        prompt += f"Q: {example['question']}\n"
        if script_args.dataset_split == "test":
            prompt += "A: "
        else:
            prompt += f"A: {example['answers'][0]}"
        return {"prompt":prompt}
    return dataset.map(make_prompt)

def preprocess_cbr_perturb(dataset, script_args):
    pass

def preprocess_dataset(dataset, script_args):
    if "cbr" in script_args.dataset_name.lower():
        dataset = preprocess_cbr(dataset, script_args)
    elif "cbr_perturb" in script_args.dataset_name.lower():
        dataset = preprocess_cbr_perturb(dataset, script_args)
    else:
        dataset = preprocess_baseline(dataset, script_args)
    return dataset

def cal_max_seq_length(dataset, script_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    max_seq_length = 0
    for i in tqdm(range(0, len(dataset), 2000), desc="Calculating max sequence length"):
        prompts = [d for d in dataset["prompt"][i:i+2000]]
        input_ids = tokenizer(prompts, return_length=True)["length"]
        max_seq_length = max(max_seq_length, max(input_ids))
    print("Max sequence length:", max_seq_length)
    return max_seq_length

def main(script_args):
    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
        if device_map == {"": 0}:
            device_map = "auto"
        torch_dtype = torch.bfloat16
        print("device_map: ", device_map)
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    print("# of GPUs: ", torch.cuda.device_count())
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        load_in_4bit=True,
        use_flash_attention_2=True,
        use_auth_token=script_args.use_auth_token,
    )

    # Step 2: Load the dataset
    dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_split)
    #TODO : preprocess 옵션 넣고, push_to_hub 옵션 넣기
    dataset = preprocess_dataset(dataset, script_args)
    print(dataset["prompt"][0])
    if script_args.cal_max_len:
        max_seq_length = cal_max_seq_length(dataset, script_args)
    else:
        max_seq_length = script_args.seq_length
    # Step 3: Define the training arguments
    run = wandb.init(
        project='Finetune-rag', 
        job_type="training",
        name=script_args.run_name,
    )
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=max_seq_length, # script_args.seq_length 사용
        train_dataset=dataset,
        dataset_text_field=script_args.dataset_text_field,
        peft_config=peft_config,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
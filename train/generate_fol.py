import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import argparse
import json
from tqdm import tqdm

def generate_fol(args):
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    device = model.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'



    ds1 = load_dataset("parquet", data_files="../datasets/full_cleaned_xsum_faith/test-00000-of-00001-00019e1f5b559f8c.parquet")["train"]
    ds2 = load_dataset("parquet", data_files="../datasets/full_cleaned_xsum_faith/train-00000-of-00001-a320c6f0065a58cc.parquet")["train"]
    ds3 = load_dataset("parquet", data_files="../datasets/full_cleaned_xsum_faith/validation-00000-of-00001-83a57f7a17701e51.parquet")["train"]

    train_dataset = concatenate_datasets([ds1, ds2, ds3])

    train_dataset = train_dataset.filter(lambda example: example["label"] == "faithful")

    train_dataset = train_dataset.map(lambda x: {"_len": len(x['document'])})
    train_dataset = train_dataset.sort("_len")
    train_dataset = train_dataset.remove_columns("_len")

    def filter_long_examples(example):
        # return len(example['document']) <= 8096
        if 'dros orllewin clwyd' in example['document']:
            return None

        return len(example['document']) <= 8096 and len(example['claim']) <= 200

    train_dataset = train_dataset.filter(filter_long_examples)

    message_prompt = ("### Instruction:\nTranslate the following natural language (NL) statement"
                       " to a first-order logic (FOL).\n\n### NL:\n{claim}\n\n### FOL:\n")
    train_dataset = train_dataset.map(lambda samples: {"prompt": message_prompt.format(**samples)})
    print(train_dataset[0])

    if os.path.exists(args.save_data_path):
        with open(args.save_data_path, 'r', encoding='utf-8') as f:
            save_list = json.load(f)
    else:
        save_list = []

    finished_count = len(save_list)
    print(f"{finished_count} completed records have been detected and will continue from {finished_count}.")

    for i in tqdm(range(finished_count, len(train_dataset), args.eval_batch_size)):
        end_idx = min(i + args.eval_batch_size, len(train_dataset))
        batch_prompts = train_dataset.select(range(i, end_idx))

        prompts = list(batch_prompts['prompt'])
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)

        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        prompt_length = inputs['input_ids'].size(1)
        completion_ids = outputs[:, prompt_length:]

        decoded_completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        result = [{'document': a, 'target_fol': b} for a, b in zip(batch_prompts['document'], decoded_completions)]
        for nl, fol in zip(batch_prompts['claim'], decoded_completions):
            print(f'claim:{nl}\nfol:{fol}\n')

        save_list.extend(result)

        with open(args.save_data_path, 'w', encoding='utf-8') as f:
            json.dump(save_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--model_name_or_path", type=str, default="../qwen2.5-1.5b--folio-sgrpo")
    parser.add_argument("--save_data_path", type=str, default="../xsum_fol_faithful_qwen2.5-1.5b-fol.json")


    args = parser.parse_args()

    generate_fol(args)

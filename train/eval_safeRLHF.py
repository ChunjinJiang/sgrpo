from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import argparse
import re
from train import parse_hha_sample_to_messages
import json
from retry import retry
import requests.exceptions
import concurrent.futures
from tqdm import tqdm
import datetime

with open("../gpt_eval.config", "r", encoding="utf-8") as f:
    config = json.load(f)
from openai import OpenAI

request_client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
system_prompt = config['system_prompt']
user_prompt = config['user_prompt']
offsetbias_prompt = config['offsetbias_prompt']
max_generation_length = 2048
log_path = "./metrics_log.txt"


def split_hha_sample(text: str):
    parts = text.strip().split("Assistant:")
    if len(parts) < 2:
        return {"instruction": text.strip(), "target": ""}

    chosen = parts[-1].strip()

    instruction = "Assistant:".join(parts[:-1]).strip()

    return {"instruction": instruction, "target": chosen}


def separate_hha(samlpe):
    message_chosen = parse_hha_sample_to_messages(samlpe['chosen'], with_label=True)
    message_rejected = parse_hha_sample_to_messages(samlpe['rejected'], with_label=True)
    message_instruction =  message_chosen[:-1]

    return {'message_instruction': message_instruction,
            'message_chosen': message_chosen,
            'message_rejected': message_rejected,}


def get_api_response(questions, answers_a, answers_b, target:str):

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Prepare task list
        futures = []
        for idx in range(len(answers_a)):
            msg = user_prompt.format(question=questions[idx], answer_a=answers_a[idx], answer_b=answers_b[idx])
            futures.append(executor.submit(request_api_for_correction, msg=msg, idx=idx, target=target))

        response = []
        for future in concurrent.futures.as_completed(futures):
            try:
                response.append(future.result())

            except Exception as e:
                print(f"request wrong: {e}")

    return response


@retry((requests.exceptions.Timeout, requests.exceptions.RequestException), tries=3, delay=10, backoff=1)
def request_api_for_correction(msg, idx, target, timeout=10):
    try:

        response = request_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ],
            response_format={'type': 'json_object'},
            stream=False,
            timeout=timeout,
        )
        content = json.loads(response.choices[0].message.content)

        return {'output': content['better_answer'], 'idx': idx, 'target': target}

    except Exception as e:
        raise RuntimeError(f"Request failed (will retry): {str(e)}")


def evaluate_hha_by_api(model_A, model_B, tokenizer_A, tokenizer_B, validation_set, batch_size, device):

    sub_metrics_win, sub_metrics_lost, sub_metrics_tie = 0, 0, 0
    size_of_dataset = len(validation_set)
    for i in tqdm(range(0, size_of_dataset, batch_size)):
        end_idx = min(i + batch_size, size_of_dataset)
        batch_prompts = validation_set.select(range(i, end_idx))

        instruction_message_with_chat_template = list(batch_prompts['prompt_with_template'])
        instruction_text = list(batch_prompts['prompt'])

        # generate completion by policy model
        inputs = tokenizer_A(instruction_message_with_chat_template, return_tensors="pt", padding=True, padding_side="left").to(device)
        generated_completion_ids = model_A.generate(**inputs, max_new_tokens=max_generation_length, pad_token_id=tokenizer_A.eos_token_id)
        prompt_length = inputs['input_ids'].size(1)
        mixed_ids = generated_completion_ids[:, prompt_length:]
        completion_text_A = tokenizer_A.batch_decode(mixed_ids, skip_special_tokens=True)

        inputs = tokenizer_B(instruction_message_with_chat_template, return_tensors="pt", padding=True, padding_side="left").to(device)
        generated_completion_ids = model_B.generate(**inputs, max_new_tokens=max_generation_length, pad_token_id=tokenizer_B.eos_token_id)
        prompt_length = inputs['input_ids'].size(1)
        mixed_ids = generated_completion_ids[:, prompt_length:]
        completion_text_B = tokenizer_B.batch_decode(mixed_ids, skip_special_tokens=True)

        staggered_position_a = get_api_response(questions=instruction_text, answers_a=completion_text_A,
                                                answers_b=completion_text_B, target='A')
        staggered_position_b = get_api_response(questions=instruction_text, answers_a=completion_text_B,
                                                answers_b=completion_text_A, target='B')
        pairwise_res = staggered_position_a + staggered_position_b

        pairwise_scoreboard = {d['idx']: 0 for d in pairwise_res}

        for res in pairwise_res:
            if res['output'] not in ['A', 'B', 'C']:
                continue

            key = res['idx']
            if res['output'] == res['target']:
                pairwise_scoreboard[key] += 1
            elif res['output'] == 'C':
                pass
            else:
                pairwise_scoreboard[key] += -1

        for item in pairwise_scoreboard.values():
            if item > 0:
                sub_metrics_win += 1
            elif item < 0:
                sub_metrics_lost += 1
            else:
                sub_metrics_tie += 1

    with open(log_path, "a", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            print(msg, file=f)

        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        subset_name = 'PKU-SafeRLHF'
        log(f"{timestamp} === Evaluation Results ({subset_name}) ===")
        log(f"Target model: {model_A.name_or_path}, Compared models: {model_B.name_or_path}")
        log(f"score/{subset_name}/api/win  {sub_metrics_win / size_of_dataset:.4f}")
        log(f"score/{subset_name}/api/lost {sub_metrics_lost / size_of_dataset:.4f}")
        log(f"score/{subset_name}/api/tie  {sub_metrics_tie / size_of_dataset:.4f}")
        log(f"Total samples: {size_of_dataset}, "
            f"Invalid sample: {size_of_dataset - (sub_metrics_win + sub_metrics_lost + sub_metrics_tie)}")
        log("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--target_model_path", type=str, default="/root/workspace/FOL-RL/Qwen3-1.7B-safeRLHF-dpo")
    parser.add_argument("--comparand_model_path", type=str, default="/root/workspace/FOL-RL/models/Qwen3-1.7B")
    parser.add_argument("--dataset_split_seed", type=int, default=0)
    parser.add_argument("--eval_data_path", type=str, default="/root/workspace/FOL-RL/datasets/PKU-SafeRLHF/PKU-SafeRLHF_default_test.parquet")

    args = parser.parse_args()

    target_model_path = args.target_model_path
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, device_map="cuda:0")
    device = target_model.device
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.padding_side = 'left'

    comparand_model_path = args.comparand_model_path
    comparand_model = AutoModelForCausalLM.from_pretrained(comparand_model_path, device_map="cuda:0")
    comparand_tokenizer = AutoTokenizer.from_pretrained(comparand_model_path)
    comparand_tokenizer.pad_token = comparand_tokenizer.eos_token
    comparand_tokenizer.padding_side = 'left'

    # ----------------------------------------------------------------------------------
    # Load PKU SafeRLHF dataset
    def to_chat_template(samples):
        text = target_tokenizer.apply_chat_template([{'role': "user", "content": samples['prompt']}],
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        better_answer_field_name = 'response_' + str(samples['better_response_id'])
        return {"prompt_with_template": text, "better_answer": samples[better_answer_field_name]}

    eval_dataset = load_dataset("parquet", data_files=args.eval_data_path)["train"]
    eval_dataset = eval_dataset.filter(lambda sample: sample['better_response_id'] == sample['safer_response_id']
                                                      and sample.get(f"is_response_{sample['better_response_id']}_safe",
                                                                     False))
    eval_dataset = eval_dataset.map(to_chat_template)
    # ----------------------------------------------------------------------------------

    evaluate_hha_by_api(model_A=target_model, model_B=comparand_model, tokenizer_A=target_tokenizer, tokenizer_B=comparand_tokenizer, validation_set=eval_dataset, device=device, batch_size=16)
    # evaluate_hha_by_offsetbias(model=model, tokenizer=tokenizer, validation_set=eval_dataset, device=device, batch_size=16)



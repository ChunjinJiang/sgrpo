import argparse

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import re


def extract_instruction(text):
    assistant_spans = [m.start() for m in re.finditer(r"\bAssistant:", text)]
    if len(assistant_spans) < 2:
        return text.strip()
    last_span = assistant_spans[-1]
    return text[:last_span].strip()


def extract_last_assistant(text: str) -> str:
    matches = re.findall(r'Assistant:\s*(.*?)(?=\nHuman:|$)', text, flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def to_hha_chat_template(samples):
    instruction_text = extract_instruction(samples['chosen'])
    chosen_text = extract_last_assistant(samples['chosen'])
    rejected_text = extract_last_assistant(samples['rejected'])

    return {"prompt": instruction_text, 'chosen': chosen_text, 'rejected': rejected_text}


def train(args):

    # === 1. Select the basic model ===
    model_name = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.generation_config.enable_thinking = False

    # hha_path = '/root/FOL-RL/datasets/hh-rlhf'
    # harmless_base = load_dataset(hha_path, data_dir="harmless-base")
    # helpful_base = load_dataset(hha_path, data_dir="helpful-base")
    # helpful_online = load_dataset(hha_path, data_dir="helpful-online")
    # helpful_rejection = load_dataset(hha_path, data_dir="helpful-rejection-sampled")
    #
    # train_dataset = concatenate_datasets([
    #     harmless_base["train"],
    #     helpful_base["train"],
    #     helpful_online["train"],
    #     helpful_rejection["train"],
    # ])
    #
    # train_dataset = train_dataset.map(to_hha_chat_template)


    train_data_path = args.dataset_path

    def to_chat_template(samples):
        text = tokenizer.apply_chat_template([{'role': "user", "content": samples['prompt']}],
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        better_answer_field_name = 'response_' + str(samples['better_response_id'])
        rejected_answer_field_name = 'response_' + str(1 - samples['better_response_id'])
        return {'prompt': text, "chosen": samples[better_answer_field_name], 'rejected': samples[rejected_answer_field_name]}


    train_dataset = load_dataset("parquet", data_files=train_data_path)["train"]
    train_dataset = train_dataset.filter(lambda sample: sample['better_response_id'] == sample['safer_response_id']
                                                        and sample.get(f"is_response_{sample['better_response_id']}_safe",
                                                                       False))
    train_dataset = train_dataset.map(to_chat_template)
    train_dataset = train_dataset.filter(lambda sample: len(sample['chosen']) > 0)

    training_args = DPOConfig(
        output_dir="./root/autodl-tmp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        bf16=False,
        # logging_steps=50,
        # save_steps=500,
        logging_strategy="epoch",
        save_strategy='no',
        num_train_epochs=1,
        remove_unused_columns=True,
        beta=0.1,
        # max_length=4096,
        # warmup_ratio=0.03,
    )


    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # Default is a frozen copy of the model
        args=training_args,
        train_dataset=train_dataset,
    )


    trainer.train()

    trainer.model.save_pretrained('./Qwen3-1.7B-safeRLHF-dpo')
    tokenizer.save_pretrained('./Qwen3-1.7B-safeRLHF-dpo')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--model_path",type=str, default="../models/Qwen3-1.7B")
    parser.add_argument("--dataset_path", type=str, default="../datasets/PKU-SafeRLHF/PKU-SafeRLHF_default_train.parquet")

    args = parser.parse_args()
    train(args)
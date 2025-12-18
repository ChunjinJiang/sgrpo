import torch
from comet import load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import transformers
from functools import partial
import argparse
from train import load_dataset_from_path, ManualEvalCallback, save_config
import wandb


def add_prompt_and_label(data_point, Tokenizer, max_input_length, GT_label, message_prompt=None):
    assert (GT_label in data_point and data_point[GT_label]), 'some datapoint is empty{}'.format(data_point)

    # prompt_input = Tokenizer.apply_chat_template([{'role':"user", "content":message_prompt.format(**data_point)}],
    #                           0                   tokenize=False,
    #                                              add_generation_prompt=True,
    #                                              enable_thinking=False)
    #
    # bos = getattr(Tokenizer, "bos_token", None)
    # if bos is not None:
    #     prompt_input = prompt_input.replace(bos, "")
    if message_prompt is None:
        prompt_input = data_point['prompt']
    else:
        prompt_input = message_prompt.format(**data_point)

    prompt_output = data_point[GT_label]

    print('test')
    prompt_input_ids = Tokenizer.encode(prompt_input, add_special_tokens=False)
    prompt_output_ids = Tokenizer.encode(prompt_output, add_special_tokens=False)

    full_input = prompt_input_ids + prompt_output_ids
    if full_input[-1] != Tokenizer.eos_token_id and len(full_input) < max_input_length:
        full_input.append(Tokenizer.eos_token_id)

    labels = [-100] * len(prompt_input_ids) + full_input[len(prompt_input_ids):]

    result = {'input_ids': full_input,
              'attention_mask': [1] * len(full_input),
              'labels': labels}

    return result


def print_sample(input_ids, labels, tokenizer):
    # decode prompt + completion
    decoded_input = tokenizer.decode(input_ids)

    label_tokens = [tok for tok in labels if tok != -100]
    decoded_label = tokenizer.decode(label_tokens)

    print("=== Prompt + Completion (from input_ids) ===")
    print(decoded_input)
    print("\n=== Completion (from labels) ===")
    print(decoded_label)


def prepare_dataset(args, tokenizer):
    message_prompt= None

    if args.task == "distill_to_fol":
        message_prompt = ("### Instruction:\nPlease describe the core meaning of the passage using first-order logic (FOL)."
                          "\n\n### Paragraph:\n{document}\n\n### FOL:\n")
        data = load_dataset("json", data_files=args.train_data_path)['train']
        train_dataset = data.filter(lambda x: len(x["document"]) >= 200)

        # sorted by len
        train_dataset = train_dataset.map(lambda x: {"_len": len(x[args.key_in_prompt])})
        train_dataset = train_dataset.sort("_len")
        train_dataset = train_dataset.remove_columns("_len")

        train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=0).values()
    else:
        train_dataset, eval_dataset = load_dataset_from_path(args, tokenizer)

    prepare_input = partial(add_prompt_and_label, Tokenizer=tokenizer, max_input_length=4096,
                            GT_label=args.GT_label, message_prompt=message_prompt)
    train_dataset = train_dataset.map(prepare_input)

    return train_dataset, eval_dataset


def train(args):
    wandb.init(project="SGRPO-FOL", name="FOL Translation", mode="offline")

    torch.cuda.empty_cache()
    wmt22_comet_da = None
    output_dir = "./sft-epoch-saved"

    print(f'model path：{args.model_name_or_path}\n'
          f'training dataset path：{args.train_data_path}\n'
          f'eval dataset path：{args.eval_data_path}\n'
          f'lr:{args.lr}\n'
          f'save path：{args.save_path}\n'
          f'dataset seed:{args.dataset_split_seed}')

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", max_memory={0: "65GiB"},)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_dataset, eval_dataset = prepare_dataset(args=args, tokenizer=tokenizer)
    # train_dataset = sorted(train_dataset, key=lambda x: len(x["better_answer"]), reverse=True)
    # train_dataset = train_dataset.select(range(50))
    # eval_dataset = {
    #     name: ds.shuffle(seed=0).select(range(250))
    #     for name, ds in eval_dataset.items()
    # }

    for i in range(4):
        print('training sample: ', train_dataset[i])
        print('eval sample(harmless-base): ', eval_dataset['harmless-base'][i])
        print('eval sample(helpful-base): ', eval_dataset['helpful-base'][i])
        print('eval sample(helpful-online): ', eval_dataset['helpful-online'][i])
        print('eval sample(helpful-rejection): ', eval_dataset['helpful-rejection'][i])
        print('----------------------------------')

    if args.task == 'wmt':
        wmt22_comet_da = load_from_checkpoint('../assistance/wmt22-comet-da/checkpoints/model.ckpt', local_files_only=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            # lr_scheduler_type='constant',
            learning_rate=args.lr,
            optim="adamw_torch",
            save_strategy="epoch",
            logging_strategy="epoch",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        ),
        # callbacks=[ManualEvalCallback(task=args.task, tokenizer=tokenizer, validation_set=eval_dataset,
        #                               batch_size=args.eval_batch_size, device=model.device, wmt22_comet_da=wmt22_comet_da,
        #                               GT_label=args.GT_label, key_in_prompt=args.key_in_prompt,
        #                               max_generation_length=args.max_generation_length)],
    )
    trainer.train()

    trainer.model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--is_sorted", type=bool, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # parser.add_argument("--max_generation_length", type=int, default=512)
    # parser.add_argument("--model_name_or_path", type=str, default="/root/workspace/FOL-RL/models/Qwen3-1.7B")
    # parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/WMT_newstest2017-2020/newstest2017-2020_en-de.json")
    # parser.add_argument("--eval_data_path", type=str, default="/root/workspace/FOL-RL/datasets/WMT22_competition/WMT22_competition_en-de.json")
    # parser.add_argument("--save_path", type=str, default="../Qwen3-1.7B-SFT-WMT-en_de")
    # parser.add_argument("--dataset_split_seed", type=int, default=0)
    # parser.add_argument("--key_in_prompt", type=str, default='src')
    # parser.add_argument("--GT_label", type=str, default='ref')
    # parser.add_argument("--task", type=str, default='wmt', choices=['distill_to_fol', 'translate_to_fol', 'wmt', 'hha'])

    # parser.add_argument("--max_generation_length", type=int, default=256)
    # parser.add_argument("--model_name_or_path", type=str, default="/root/workspace/FOL-RL/models/Qwen3-1.7B")
    # parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/FOLIO/folio_with_steps.json")
    # parser.add_argument("--eval_data_path", type=str, default=None)
    # parser.add_argument("--save_path", type=str, default="../Qwen3-1.7B-SFT-fol")
    # parser.add_argument("--dataset_split_seed", type=int, default=0)
    # parser.add_argument("--key_in_prompt", type=str, default='NL')
    # parser.add_argument("--GT_label", type=str, default='FOL')
    # parser.add_argument("--task", type=str, default='translate_to_fol', choices=['distill_to_fol', 'translate_to_fol', 'wmt', 'hha'])

    parser.add_argument("--max_generation_length", type=int, default=2048)
    parser.add_argument("--model_name_or_path", type=str, default="/root/workspace/FOL-RL/models/Qwen3-1.7B")
    parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/hh-rlhf")
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="../Qwen3-1.7B-SFT-HHA")
    parser.add_argument("--dataset_split_seed", type=int, default=0)
    parser.add_argument("--key_in_prompt", type=str, default=None)
    parser.add_argument("--GT_label", type=str, default='better_answer')
    parser.add_argument("--task", type=str, default='hha', choices=['distill_to_fol', 'translate_to_fol', 'wmt', 'hha'])

    args = parser.parse_args()

    train(args)
    save_config(args)

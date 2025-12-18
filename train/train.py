import datetime

import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from transformers import TrainerCallback

from utils.sgrpo_trainer import SuperviseGRPOTrainer
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
from itertools import groupby
from comet import load_from_checkpoint
from sacrebleu.metrics import BLEU
import sacrebleu
from utils.LogicalEquivalence import Compute_LE, msplit
import wandb
import json
from tqdm import tqdm
from retry import retry
import requests.exceptions
import concurrent.futures


FOL_tokenizer = lambda x: msplit(x)[0]

def compute_fol_bleu_batch(pred_fols, true_fols, tokenizer=FOL_tokenizer):
    """
    Implement FOL BLEU calculation equivalent to evaluate.load ('breu ') using sacrebreu.
    Automatically adjust the effective n-gram order based on the shortest token length (simulating max_order).
    """
    assert len(pred_fols) == len(true_fols), "Inconsistent prediction and reference quantity"

    # tokenize
    preds_tok = [" ".join(tokenizer(p)) for p in pred_fols]
    refs_tok = [[" ".join(tokenizer(r))] for r in true_fols]

    # Automatically determine n-gram order
    # min_len = min(len(tokenizer(p)) for p in pred_fols + true_fols)
    # max_order = min(4, min_len)

    bleu = sacrebleu.corpus_bleu(
        preds_tok,
        refs_tok,
        smooth_method='exp',
        smooth_value=None,
        tokenize='none',
        use_effective_order=True
    )

    # precisions = bleu.precisions[:max_order]
    # precisions += [0.0] * (4 - len(precisions))

    # return {'bleu': bleu.score / 100.0, 'precisions': [p / 100.0 for p in precisions],
    #         'brevity_penalty': bleu.bp, 'max_order': max_order}
    return bleu.score / 100.0


def normalize_to_unit_interval(values: list[float]) -> list[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [0.0 for _ in values]
    return [(x - min_val) / (max_val - min_val) for x in values]


class ManualEvalCallback(TrainerCallback):
    def __init__(self, task, tokenizer, validation_set,
                 wmt22_comet_da, batch_size, GT_label, key_in_prompt, device,
                 max_generation_length, hha_rm_model=None, hha_rm_tokenizer=None, hha_rm_type=None):
        self.task = task
        self.tokenizer = tokenizer
        self.device = device
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.GT_label = GT_label
        self.key_in_prompt = key_in_prompt
        self.wmt22_comet_da = wmt22_comet_da
        self.max_generation_length = max_generation_length

        self.hha_rm_model = hha_rm_model
        self.hha_rm_tokenizer = hha_rm_tokenizer
        self.hha_rm_type = hha_rm_type

        if task == 'hha':
            with open("../hha_gpt_eval.config", "r", encoding="utf-8") as f:
                config = json.load(f)
            from openai import OpenAI
            self.request_client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
            self.system_prompt = config['system_prompt']
            self.user_prompt = config['user_prompt']
            self.offsetbias_prompt = config['offsetbias_prompt']

    def get_api_response(self, questions, answers_a, answers_b, target: str):

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 准备任务列表
            futures = []
            for idx in range(len(answers_a)):
                msg = self.user_prompt.format(question=questions[idx], answer_a=answers_a[idx], answer_b=answers_b[idx])
                futures.append(executor.submit(self.request_api_for_correction, msg=msg, idx=idx, target=target))

            response = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    response.append(future.result())

                except Exception as e:
                    print(f"request wrong: {e}")

        return response
        return response

    @retry((requests.exceptions.Timeout, requests.exceptions.RequestException), tries=3, delay=10, backoff=1)
    def request_api_for_correction(self, msg, idx, target, timeout=10):
        try:

            response = self.request_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": self.system_prompt},
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


    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        model.eval()
        with torch.no_grad():
            epoch = int(state.epoch)

            if self.task == "translate_to_fol":
                metrics = self.evaluate_fol_fn(model=model, tokenizer=self.tokenizer,
                                               validation_set=self.validation_set,
                                               batch_size=self.batch_size, device=self.device, GT_label=self.GT_label)
                print({"score/LE": metrics["LE"], "epoch": epoch})
                print({"score/bleu": metrics["bleu"], "epoch": epoch})
                wandb.log({"score/LE": metrics["LE"], "epoch": epoch})
                wandb.log({"score/bleu": metrics["bleu"], "epoch": epoch})

            elif self.task == "wmt":
                 metrics = self.evaluate_wmt_fn(model=model, tokenizer=self.tokenizer, validation_set=self.validation_set,
                                               batch_size=self.batch_size, device=self.device,
                                                GT_label=self.GT_label, key_in_prompt=self.key_in_prompt,
                                                max_generation_length=self.max_generation_length,
                                                wmt22_comet_da=self.wmt22_comet_da)
                 wandb.log({"score/En=>De_bleu": metrics["en2de_bleu"], "epoch": epoch})
                 wandb.log({"score/En=>De_comet": metrics["en2de_comet"], "epoch": epoch})
                 wandb.log({"score/De=>En_bleu": metrics["de2en_bleu"], "epoch": epoch})
                 wandb.log({"score/De=>En_comet": metrics["de2en_comet"], "epoch": epoch})
            elif self.task == "hha":
                self.evaluate_hha_by_api(model=model, tokenizer=self.tokenizer,
                                     validation_set=self.validation_set,
                                     batch_size=self.batch_size, device=self.device, epoch=epoch)

                self.evaluate_hha_by_offsetbias(model=model, tokenizer=self.tokenizer,
                                                validation_set=self.validation_set,
                                                batch_size=self.batch_size, device=self.device, epoch=epoch)
            else:
                raise ValueError(f"Unknown task: {self.task}")

        model.train()

        return control


    def evaluate_hha_by_offsetbias(self, model, tokenizer, validation_set, batch_size, device, epoch):
        hha_rm_tokenizer = AutoTokenizer.from_pretrained('../assistance/Llama-3-OffsetBias-8B', )
        hha_rm_tokenizer.pad_token = hha_rm_tokenizer.eos_token
        hha_rm_tokenizer.padding_side = "left"
        hha_rm_model = AutoModelForCausalLM.from_pretrained('../assistance/Llama-3-OffsetBias-8B',
                                                            device_map="cuda:0")

        for subset_name, eval_sub_dataset in validation_set.items():
            eval_sub_dataset = eval_sub_dataset.map(self.separate_hha)

            sub_metrics_win, sub_metrics_lost, sub_metrics_tie = 0, 0, 0
            size_of_dataset = len(eval_sub_dataset)
            for i in range(0, size_of_dataset, batch_size):
                end_idx = min(i + batch_size, size_of_dataset)
                batch_prompts = eval_sub_dataset.select(range(i, end_idx))

                # didn't actually get the chosen, just extracted the instructions from it
                instrution_chosen_text = batch_prompts['chosen']
                chosen_text = batch_prompts['better_answer']

                instruction_text = [instruction[:-len(chosen)] for instruction, chosen in zip(instrution_chosen_text, chosen_text)]

                instruction_message_with_chat_template = [
                    tokenizer.apply_chat_template(
                        parse_hha_sample_to_messages(sample),
                        tokenize=False,
                        add_generation_prompt=True
                    ).replace(tokenizer.bos_token, "")
                    for sample in instruction_text
                ]

                # generate completion by policy model
                inputs = tokenizer(instruction_message_with_chat_template, return_tensors="pt", padding_side="left").to(device)
                generated_completion_ids = model.generate(**inputs, max_new_tokens=self.max_generation_length, pad_token_id=tokenizer.eos_token_id)
                prompt_length = inputs['input_ids'].size(1)
                mixed_ids = generated_completion_ids[:, prompt_length:]

                completion_text = tokenizer.batch_decode(mixed_ids, skip_special_tokens=True)

                user_message = [{"role": "user", "content": self.offsetbias_prompt.format(input=ins, output_1=target, output_2=chosen)}
                                for ins, target, chosen in zip(instruction_text, completion_text, chosen_text)]

                user_message_reverse = [{"role": "user", "content": self.offsetbias_prompt.format(input=ins, output_1=chosen, output_2=target)}
                                for ins, target, chosen in zip(instruction_text, completion_text, chosen_text)]

                mixed_message = user_message + user_message_reverse
                input_judge = [tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt")
                    for conversation in mixed_message
                ]

                input_judge_ids = tokenizer(input_judge, return_tensors="pt", padding=True, padding_side="left").to(device)
                res_judge = hha_rm_model.generate(
                    input_ids=input_judge_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=128009,
                    temperature=0)

                prompt_length = input_judge_ids['input_ids'].size(1)
                mixed_ids = res_judge[:, prompt_length:]

                pairwise_res = tokenizer.batch_decode(mixed_ids, skip_special_tokens=True)
                pairwise_scoreboard = {str(i): 0 for i in range(len(user_message))}

                for res in pairwise_res:
                    if res['output'] not in ['(a)', '(b)', '(c)']:
                        continue

                    key = str(res['idx'])
                    if res['output'] == res['target']:
                        pairwise_scoreboard[key] += 1
                    elif res['output'] == '[C]':
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
            wandb.log({"score/" + subset_name + "/offsetbias/win": sub_metrics_win / size_of_dataset, "epoch": epoch})
            wandb.log({"score/" + subset_name + "/offsetbias/lost": sub_metrics_lost / size_of_dataset, "epoch": epoch})
            wandb.log({"score/" + subset_name + "/offsetbias/tie": sub_metrics_tie / size_of_dataset, "epoch": epoch})


    def evaluate_hha_by_api(self, model, tokenizer, validation_set, batch_size, device, epoch):

        for subset_name, eval_sub_dataset in validation_set.items():

            sub_metrics_win, sub_metrics_lost, sub_metrics_tie = 0, 0, 0
            size_of_dataset = len(eval_sub_dataset)
            for i in tqdm(range(0, size_of_dataset, batch_size), desc=subset_name):
                end_idx = min(i + batch_size, size_of_dataset)
                batch_prompts = eval_sub_dataset.select(range(i, end_idx))

                instruction_text, chosen_text = [], []
                for sample in batch_prompts["chosen"]:
                    res = self.split_hha_sample(sample)
                    instruction_text.append(res["instruction"])
                    chosen_text.append(res["target"])

                instruction_message_with_chat_template = [
                    tokenizer.apply_chat_template(
                        parse_hha_sample_to_messages(sample),
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for sample in instruction_text
                ]

                # generate completion by policy model
                inputs = tokenizer(instruction_message_with_chat_template, return_tensors="pt", padding=True,
                                   padding_side="left").to(device)
                generated_completion_ids = model.generate(**inputs, max_new_tokens=self.max_generation_length,
                                                          pad_token_id=tokenizer.eos_token_id)
                prompt_length = inputs['input_ids'].size(1)
                mixed_ids = generated_completion_ids[:, prompt_length:]

                completion_text = tokenizer.batch_decode(mixed_ids, skip_special_tokens=True)
                staggered_position_a = self.get_api_response(questions=instruction_text, answers_a=chosen_text,
                                                        answers_b=completion_text, target='B')
                staggered_position_b = self.get_api_response(questions=instruction_text, answers_a=completion_text,
                                                        answers_b=chosen_text, target='A')
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

            with open('hha_eval.log', "a", encoding="utf-8") as f:
                def log(msg):
                    print(msg)
                    print(msg, file=f)

                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log(f"{timestamp} === Evaluation Results ({subset_name}) ===")
                log(f"score/{subset_name}/api/win  {sub_metrics_win / size_of_dataset:.4f}")
                log(f"score/{subset_name}/api/lost {sub_metrics_lost / size_of_dataset:.4f}")
                log(f"score/{subset_name}/api/tie  {sub_metrics_tie / size_of_dataset:.4f}")
                log(f"Total samples: {size_of_dataset}, "
                    f"Invalid sample: {size_of_dataset - (sub_metrics_win + sub_metrics_lost + sub_metrics_tie)}")
                log("-" * 60)


    @staticmethod
    def evaluate_fol_fn(model, tokenizer, validation_set, batch_size, device, GT_label):

        sum_LE_Jiang, sum_f1, sum_bleu = 0.0, 0.0, 0.0
        for i in range(0, len(validation_set), batch_size):
            end_idx = min(i + batch_size, len(validation_set))
            batch_prompts = validation_set.select(range(i, end_idx))

            prompts = list(batch_prompts['prompt'])
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)

            outputs = model.generate(**inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
            prompt_length = inputs['input_ids'].size(1)
            completion_ids = outputs[:, prompt_length:]

            decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            bleu_score = compute_fol_bleu_batch(pred_fols=decoded, true_fols=batch_prompts['FOL'])
            sum_bleu = sum_bleu + bleu_score * (end_idx - i)

            for j in range(end_idx - i):
                # compute f1 score
                # pred_ids = tokenizer.encode(decoded[j], add_special_tokens=False)
                # GT_ids = tokenizer.encode(batch_prompts[j][GT_label], add_special_tokens=False)
                #
                # set_pred = set(pred_ids)
                # set_corrected = set(GT_ids)
                # common = len(set_pred & set_corrected)
                # precision = common / len(set_pred) if set_pred else 0
                # recall = common / len(set_corrected) if set_corrected else 0
                # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                # sum_f1 = sum_f1 + f1

                if decoded[j].count('∧') > 5 or decoded[j].count('∨') > 5:
                    continue

                _, singleLE_Jiang = Compute_LE(pred_text_FOL=decoded[j], true_text_FOL=batch_prompts[j]['FOL'])

                sum_LE_Jiang = sum_LE_Jiang + singleLE_Jiang[0]

        # return {'LE': sum_LE_Jiang / len(validation_set), 'f1': sum_f1 / len(validation_set)}
        return {'LE': sum_LE_Jiang / len(validation_set), 'bleu': sum_bleu / len(validation_set)}


    @staticmethod
    def evaluate_wmt_fn(model, tokenizer, validation_set, batch_size,
                        device, key_in_prompt, GT_label, max_generation_length, wmt22_comet_da):

        sum_en2de_bleu_score, sum_en2de_wmt22_comet_da = 0.0, 0.0
        sum_de2en_bleu_score, sum_de2en_wmt22_comet_da = 0.0, 0.0
        for i in tqdm(range(0, len(validation_set), batch_size), desc="evaluation"):
            # TODO:确认分数都是正确的
            end_idx = min(i + batch_size, len(validation_set))
            batch_dataset = validation_set.select(range(i, end_idx))

            prompts = list(batch_dataset['prompt'])
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)

            outputs = model.generate(**inputs, max_new_tokens=max_generation_length,
                                     pad_token_id=tokenizer.eos_token_id)
            prompt_length = inputs['input_ids'].size(1)
            completion_ids = outputs[:, prompt_length:]

            output_translate = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            refence_translate = validation_set[i:end_idx][GT_label]
            srcs_language = validation_set[i:end_idx][key_in_prompt]

            bleu = BLEU()
            # ------------------------------------------------------
            en2de_ouput = [item for i, item in enumerate(output_translate) if i % 2 == 0]
            en2de_ref = [item for i, item in enumerate(refence_translate) if i % 2 == 0]
            en2de_src = [item for i, item in enumerate(srcs_language) if i % 2 == 0]
            sum_en2de_bleu_score = sum_en2de_bleu_score + bleu.corpus_score(en2de_ouput, [en2de_ref]).score * len(
                en2de_ouput)

            input_for_comet = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(en2de_src, en2de_ouput, en2de_ref)
            ]
            output_comet = wmt22_comet_da.predict(input_for_comet, batch_size=batch_size, progress_bar=False,
                                                  num_workers=0, gpus=1)
            sum_en2de_wmt22_comet_da = sum_en2de_wmt22_comet_da + output_comet['system_score'] * len(en2de_ouput)
            # ------------------------------------------------------
            de2en_ouput = [item for i, item in enumerate(output_translate) if i % 2 == 1]
            de2en_ref = [item for i, item in enumerate(refence_translate) if i % 2 == 1]
            de2en_src = [item for i, item in enumerate(srcs_language) if i % 2 == 1]
            sum_de2en_bleu_score = sum_de2en_bleu_score + bleu.corpus_score(de2en_ouput, [de2en_ref]).score * len(
                de2en_ouput)

            input_for_comet = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(de2en_src, de2en_ouput, de2en_ref)
            ]
            output_comet = wmt22_comet_da.predict(input_for_comet, batch_size=batch_size, progress_bar=False,
                                                  num_workers=0, gpus=1)
            sum_de2en_wmt22_comet_da = sum_de2en_wmt22_comet_da + output_comet['system_score'] * len(de2en_ouput)
            # ------------------------------------------------------

        return {'en2de_bleu': sum_en2de_bleu_score / len(validation_set) * 2,
                'en2de_comet': sum_en2de_wmt22_comet_da / len(validation_set) * 2,
                'de2en_bleu': sum_de2en_bleu_score / len(validation_set) * 2,
                'de2en_comet': sum_de2en_wmt22_comet_da / len(validation_set) * 2}


    @staticmethod
    def split_hha_sample(text: str):
        parts = text.strip().split("Assistant:")
        if len(parts) < 2:
            return {"instruction": text.strip(), "target": ""}

        chosen = parts[-1].strip()

        instruction = "Assistant:".join(parts[:-1]).strip()

        return {"instruction": instruction, "target": chosen}


    @staticmethod
    def separate_hha(samlpe):
        message_chosen = parse_hha_sample_to_messages(samlpe['chosen'], with_label=True)
        message_rejected = parse_hha_sample_to_messages(samlpe['rejected'], with_label=True)
        message_instruction =  message_chosen[:-1]

        return {'message_instruction': message_instruction,
                'message_chosen': message_chosen,
                'message_rejected': message_rejected,}



def make_reward_func(key_in_prompt, task, GT_label, tokenizer=None, wmt22_comet_da=None, hha_rm_model=None, hha_rm_tokenizer=None, hha_rm_type=None):

    def reward_func_wmt(key_in_prompt=key_in_prompt, wmt22_comet_da=wmt22_comet_da, **kwargs):

        output_translate = kwargs['completions']
        refence_translate = kwargs[GT_label]
        srcs_language = kwargs[key_in_prompt]

        bleu_score_list = []
        bleu = BLEU(effective_order=True)
        for hypothesis, reference in zip(output_translate, refence_translate):
            bleu_score = bleu.sentence_score(hypothesis, [reference])
            bleu_score_list.append(bleu_score.score / 100.0)

        input_for_comet = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(srcs_language, output_translate, refence_translate)
        ]
        output_comet = wmt22_comet_da.predict(input_for_comet, batch_size=len(output_translate), progress_bar=False)
        comet_score_list = output_comet['scores']

        reward = [(a + b) * 0.5 for a, b in zip(bleu_score_list, comet_score_list)]

        return torch.tensor(reward)

    def reward_func_fol(tokenizer=tokenizer, **kwargs):
        # Calculate F1 using encoded tokens
        predicted_completion_ids = kwargs['completion_ids']
        corrected_ids = tokenizer(text=kwargs[GT_label], add_special_tokens=False, return_tensors=None)[
            'input_ids']

        reward = []
        for i in range(len(corrected_ids)):
            if len(kwargs['completions'][i]) > 180:
                reward.append(0.0)
                continue

            # set_pred = set(predicted_completion_ids[i])
            # set_corrected = set(corrected_ids[i])
            # common = len(set_pred & set_corrected)
            # precision = common / len(set_pred) if set_pred else 0
            # recall = common / len(set_corrected) if set_corrected else 0

            preds_tok = " ".join(FOL_tokenizer(kwargs['completions'][i]))
            refs_tok = [" ".join(FOL_tokenizer(kwargs[GT_label][i]))]

            bleu = sacrebleu.corpus_bleu(
                preds_tok,
                refs_tok,
                smooth_method='exp',
                smooth_value=None,
                tokenize='none',
                use_effective_order=True
            )

            _, singleLE_Jiang = Compute_LE(pred_text_FOL=kwargs['completions'][i],
                                           true_text_FOL=kwargs[GT_label][i])

            score = 0.5 * (singleLE_Jiang[0] + bleu.score / 100.0)
            reward.append(score)

        return torch.tensor(reward)

    def reward_func_hha_by_offsetbias(hha_rm_tokenizer=hha_rm_tokenizer, hha_rm_model=hha_rm_model, GT_label=GT_label, **kwargs):

        prompt = kwargs[GT_label]
        completions = kwargs['completions']

        prompt_with_chat_template = [
            hha_rm_tokenizer.apply_chat_template(
                parse_hha_sample_to_messages(sample),
                tokenize=False,
                add_generation_prompt=True
            ).replace(hha_rm_tokenizer.bos_token, "")
            for sample in prompt
        ]
        prompt_completion = [x + y for x, y in zip(prompt_with_chat_template, completions)]

        pipe_outputs = hha_rm_model(prompt_completion, **{"return_all_scores": True,
                                                         "function_to_apply": "none", "batch_size": 1})
        # TODO:检查输出的分数数量对不对
        reward = [item['score'] for item in pipe_outputs]
        normalized_reward = normalize_to_unit_interval(reward)

        return torch.tensor(normalized_reward)

    # def reward_func_hha_by_fol(hha_rm_tokenizer=hha_rm_tokenizer, hha_rm_model=hha_rm_model, key_in_prompt=key_in_prompt, **kwargs):
    #
    #     from utils.LogicalEquivalence import Compute_LE
    #
    #     device = hha_rm_model.device
    #
    #     completions = kwargs['completions']
    #     refences = kwargs['label']
    #
    #     question_prompt = ("### Instruction:\nTranslate the following natural language (NL) statement"
    #                        " to a first-order logic (FOL).\n\n### NL:\n{NL}\n\n### FOL:\n")
    #
    #     completions = [question_prompt.format(NL=item) for item in completions]
    #
    #     inputs_completion = hha_rm_tokenizer(completions, return_tensors="pt", padding=True, padding_side="left").to(device)
    #     fol_completions = hha_rm_model.generate(**inputs_completion, max_new_tokens=512, pad_token_id=hha_rm_tokenizer.eos_token_id)
    #     prompt_length = inputs_completion['input_ids'].size(1)
    #     comp_ids = fol_completions[:, prompt_length:]
    #
    #     fol_completions = hha_rm_tokenizer.batch_decode(comp_ids)
    #
    #     compressed_ref = [k for k, _ in groupby(refences)]
    #     compressed_ref = [question_prompt.format(NL=item) for item in compressed_ref]
    #     inputs_ref = hha_rm_tokenizer(compressed_ref, return_tensors="pt", padding=True, padding_side="left").to(device)
    #     fol_ref = hha_rm_model.generate(**inputs_ref, max_new_tokens=512, pad_token_id=hha_rm_tokenizer.eos_token_id)
    #     prompt_length = inputs_ref['input_ids'].size(1)
    #     ref_ids = fol_ref[:, prompt_length:]
    #
    #     fol_ref = hha_rm_tokenizer.batch_decode(ref_ids)
    #     fol_dict = dict(zip(compressed_ref, fol_ref))
    #
    #     reward = [Compute_LE(pred_text_FOL=fol_com, true_text_FOL=fol_dict[ref]) for fol_com, ref in zip(fol_completions, refences)]
    #
    #     return torch.tensor(reward)


    def reward_func_hha_by_fol(hha_rm_tokenizer=hha_rm_tokenizer, hha_rm_model=hha_rm_model, **kwargs):

        from utils.LogicalEquivalence import Compute_LE

        device = hha_rm_model.device

        completions = kwargs['completions']
        refences = kwargs['better_answer']

        question_prompt = ("### Instruction:\nPlease describe the core meaning of the passage using first-order logic (FOL)."
                          "\n\n### Paragraph:\n{document}\n\n### FOL:\n")

        compressed_ref = [k for k, _ in groupby(refences)]

        # Separation after superposition to generate FOL results
        mixed_input = completions + compressed_ref
        mixed_input = [question_prompt.format(document=item) for item in mixed_input]

        inputs = hha_rm_tokenizer(mixed_input, return_tensors="pt", padding=True, padding_side="left").to(device)
        fol_mixed = hha_rm_model.generate(**inputs, max_new_tokens=256, pad_token_id=hha_rm_tokenizer.eos_token_id)
        prompt_length = inputs['input_ids'].size(1)
        mixed_ids = fol_mixed[:, prompt_length:]

        fol_mixed = hha_rm_tokenizer.batch_decode(mixed_ids, skip_special_tokens=True)

        fol_ref = fol_mixed[len(completions):]
        fol_completions = fol_mixed[:len(completions)]

        fol_dict = dict(zip(compressed_ref, fol_ref))

        reward = [Compute_LE(pred_text_FOL=fol_com, true_text_FOL=fol_dict[ref], match_with_threshold=True)[1][0] for fol_com, ref in
                  zip(fol_completions, refences)]

        return torch.tensor(reward)

    if task == 'translate_to_fol':
        return reward_func_fol
    elif task == 'wmt':
        return reward_func_wmt
    elif task == 'preference' and hha_rm_type == 'fol':
        return reward_func_hha_by_fol
    elif task == 'preference' and hha_rm_type == 'offsetbias':
        return reward_func_hha_by_offsetbias
    else:
        raise ValueError(f"Unknown task type: {task}.")


def make_bidirectional_wmt(dataset: Dataset, key_in_prompt, GT_label):

    # ----------------------------- English to German ------------------------------------
    prompt_en2de = "Translate the following English into German: {src}\n"
    processed_dataset_en2de = dataset.map(lambda samples: {"prompt": prompt_en2de.format(**samples)}).shuffle(seed=0)

    # sorted by length
    processed_dataset_en2de = processed_dataset_en2de.map(lambda x: {"_len": len(x[key_in_prompt])})
    processed_dataset_en2de = processed_dataset_en2de.sort("_len")
    processed_dataset_en2de = processed_dataset_en2de.remove_columns("_len")
    # --------------------------- German to English --------------------------------------
    prompt_de2en = "Bitte übersetzen Sie das folgende Deutsche ins Englische: {src}\n"

    dataset_de2en = dataset.map(lambda x: {key_in_prompt: x[GT_label], GT_label: x[key_in_prompt]})
    processed_dataset_de2en = dataset_de2en.map(lambda samples: {"prompt": prompt_de2en.format(**samples)}).shuffle(seed=0)

    # sorted by length
    processed_dataset_de2en = processed_dataset_de2en.map(lambda x: {"_len": len(x[GT_label])})
    processed_dataset_de2en = processed_dataset_de2en.sort("_len")
    processed_dataset_de2en = processed_dataset_de2en.remove_columns("_len")
    # ------------------------------ done -----------------------------------

    interleaved = []
    for i in range(len(processed_dataset_de2en)):
        interleaved.append(processed_dataset_en2de[i])
        interleaved.append(processed_dataset_de2en[i])

    train_dataset = Dataset.from_list(interleaved)
    return train_dataset


def parse_hha_sample_to_messages(sample: str, with_label: bool=False):
    """
    Decompose HHA samples into multiple rounds of dialogue messages
    """
    # 按 Human / Assistant 分割
    turns = re.split(r"(Human:|Assistant:)", sample)
    turns = [t.strip() for t in turns if t.strip()]

    messages = []
    role_map = {"Human:": "user", "Assistant:": "assistant"}

    current_role = None
    for t in turns:
        if t in role_map:
            current_role = role_map[t]
        else:
            if current_role is None:
                raise ValueError("Analysis error: Role not found")
            messages.append({"role": current_role, "content": t})

    # 如果是生成任务 -> 删除最后一条 assistant 回复
    if not with_label and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    return messages


def extract_instruction(text):
    assistant_spans = [m.start() for m in re.finditer(r"\bAssistant:", text)]
    if len(assistant_spans) < 2:
        return text.strip()
    last_span = assistant_spans[-1]
    return text[:last_span].strip()


def load_dataset_from_path(args, tokenizer):

    if args.task == 'hha':

        harmless_base = load_dataset(args.train_data_path, data_dir="harmless-base")
        helpful_base = load_dataset(args.train_data_path, data_dir="helpful-base")
        helpful_online = load_dataset(args.train_data_path, data_dir="helpful-online")
        helpful_rejection = load_dataset(args.train_data_path, data_dir="helpful-rejection-sampled")

        # 2. 拼接训练集（不区分）
        train_dataset = concatenate_datasets([
            harmless_base["train"],
            helpful_base["train"],
            helpful_online["train"],
            helpful_rejection["train"],
        ])

        def to_hha_chat_template(samples):
            message = parse_hha_sample_to_messages(samples['chosen'], with_label=True)

            true_label = message[-1]['content']
            _message = message[:-1]
            prompt_text = tokenizer.apply_chat_template(_message, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)

            return {"prompt": prompt_text, 'better_answer': true_label}

        train_dataset.shuffle(seed=0)

        """
        map to below:
        sample:{'chosen':original chosen,
         'rejected': original rejected,
         'prompt': instruction with chat template for trainer,
          'label': chosen without prompt}
        """
        train_dataset = train_dataset.map(to_hha_chat_template)

        eval_dataset = {
            "harmless-base": harmless_base["test"].map(to_hha_chat_template),
            "helpful-base": helpful_base["test"].map(to_hha_chat_template),
            "helpful-online": helpful_online["test"].map(to_hha_chat_template),
            "helpful-rejection": helpful_rejection["test"].map(to_hha_chat_template),
        }
    elif args.task == 'preference':
        # 加载PKU-SafeRLHF数据集作为训练集
        def to_chat_template(samples):
            text = tokenizer.apply_chat_template([{'role': "user", "content": samples['prompt']}],
                                                 tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            better_answer_field_name = 'response_'+ str(samples['better_response_id'])
            return {"prompt": text, "better_answer": samples[better_answer_field_name]}

        train_dataset = load_dataset("parquet", data_files=args.train_data_path)["train"]
        train_dataset = train_dataset.filter(lambda sample: sample['better_response_id'] == sample['safer_response_id']
                                             and sample.get(f"is_response_{sample['better_response_id']}_safe", False))
        train_dataset = train_dataset.map(to_chat_template)
        train_dataset = train_dataset.filter(lambda sample: len(sample['better_answer']) > 0)
        # train_dataset = train_dataset.filter(lambda sample: 0 < len(sample['better_answer']) < 4096)

        eval_dataset = load_dataset("parquet", data_files=args.eval_data_path)["train"]
        eval_dataset = eval_dataset.filter(lambda sample: sample['better_response_id'] == sample['safer_response_id']
                                             and sample.get(f"is_response_{sample['better_response_id']}_safe", False))
        eval_dataset = eval_dataset.map(to_chat_template)
        eval_dataset = eval_dataset.filter(lambda sample: len(sample['better_answer']) > 0)

    elif args.task == 'wmt':
        train_dataset = load_dataset("json", data_files=args.train_data_path)
        train_dataset = train_dataset['train']
        train_dataset = make_bidirectional_wmt(train_dataset, key_in_prompt=args.key_in_prompt, GT_label=args.GT_label)

        eval_dataset = load_dataset("json", data_files=args.eval_data_path)['train']
        eval_dataset = make_bidirectional_wmt(eval_dataset, key_in_prompt=args.key_in_prompt, GT_label=args.GT_label)

    elif args.task == 'translate_to_fol':
        train_dataset = load_dataset("json", data_files=args.train_data_path)
        question_prompt = ("### Instruction:\nTranslate the following natural language (NL) statement"
                           " to a first-order logic (FOL).\n\n### NL:\n{NL}\n\n### FOL:\n")

        train_dataset = train_dataset['train']
        train_dataset = train_dataset.map(lambda samples: {"prompt": question_prompt.format(**samples)})

        # 记得删
        train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=2).values()
        # eval_dataset = None

        train_dataset = train_dataset.map(lambda x: {"_len": len(x[args.GT_label])})
        train_dataset = train_dataset.sort("_len")
        train_dataset = train_dataset.remove_columns("_len")

    else:
        raise ValueError(f"Unknown task type: {args.task}.")

    return train_dataset, eval_dataset


def save_config(args):
    args_dict = vars(args)
    with open(args.save_path + '/training_config.json', "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)


def train(args):
    wandb.init(project="SGRPO", name="FOL Translation", id="test3", resume="must")
    # wandb.init(project="SGRPO-FOL", name="FOL Translation", mode="offline")

    wmt22_comet_da = None
    hha_rm_model = None
    hha_rm_tokenizer = None

    model_path = args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    device = model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset, eval_dataset = load_dataset_from_path(args, tokenizer=tokenizer)
    print('Sample data:', train_dataset[0])
    # train_dataset = train_dataset.select(range(30))

    if args.task == 'wmt':
        wmt22_comet_da = load_from_checkpoint('../assistance/wmt22-comet-da/checkpoints/model.ckpt', local_files_only=True)

    if args.reward_type == 'fol':
        hha_rm_tokenizer = AutoTokenizer.from_pretrained('../assistance/Qwen3-1.7B-sft-xsum_folio_distill_to_fol')
        hha_rm_tokenizer.pad_token = hha_rm_tokenizer.eos_token
        hha_rm_tokenizer.padding_side = "left"

        hha_rm_model =  AutoModelForCausalLM.from_pretrained('../assistance/Qwen3-1.7B-sft-xsum_folio_distill_to_fol', device_map="cuda:0")
    elif args.reward_type == 'offsetbias':
        hha_rm_tokenizer = AutoTokenizer.from_pretrained('../assistance/Llama-3-OffsetBias-RM-8B',)
        hha_rm_tokenizer.pad_token = hha_rm_tokenizer.eos_token
        hha_rm_tokenizer.padding_side = "left"

        hha_rm_model =  AutoModelForCausalLM.from_pretrained('../assistance/Llama-3-OffsetBias-RM-8B', device_map="cuda:0")

    reward_func = make_reward_func(tokenizer=tokenizer, key_in_prompt=args.key_in_prompt,
                                   wmt22_comet_da=wmt22_comet_da, task=args.task, GT_label=args.GT_label,
                                   hha_rm_model=hha_rm_model, hha_rm_tokenizer=hha_rm_tokenizer, hha_rm_type=args.reward_type)

    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False
    training_args = GRPOConfig(
        output_dir="./sgrpo_epoch_saved",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        logging_strategy="epoch",
        save_strategy="steps", # epoch
        save_steps=50000,  # Save every 50000 steps
        save_total_limit=100,  # Keep up to 100 checkpoints, old ones will be automatically deleted
        num_train_epochs=args.epochs,
        max_completion_length=args.max_generation_length,
        bf16=args.bf16,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        shuffle_dataset=False,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # gradient_checkpointing=True
    )

    callbacks = []
    if args.reward_type is None:
        callbacks.append(ManualEvalCallback(task=args.task, tokenizer=tokenizer, validation_set=eval_dataset,
                                      batch_size=args.eval_batch_size, device=device, wmt22_comet_da=wmt22_comet_da,
                                      GT_label=args.GT_label, key_in_prompt=args.key_in_prompt,
                                      max_generation_length=args.max_generation_length))
    trainer = SuperviseGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_func],
        GT_label=args.GT_label,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # save model
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.12)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--bf16", type =bool, default=True)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="../saved_model")
    parser.add_argument("--model_name_or_path", type=str, default="/root/workspace/FOL-RL/models/Qwen3-1.7B")
    # parser.add_argument("--model_name_or_path", type=str, default="/root/workspace/FOL-RL/Qwen3-1.7B-sft-folio-seed2")

    parser.add_argument("--task", type=str, default='preference', choices=['translate_to_fol', 'wmt', 'preference'])
    parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/PKU-SafeRLHF/PKU-SafeRLHF_default_train.parquet")
    parser.add_argument("--eval_data_path", type=str, default="/root/workspace/FOL-RL/datasets/PKU-SafeRLHF/PKU-SafeRLHF_default_test.parquet")
    parser.add_argument("--reward_type", type=str, default='fol', choices=['fol', 'offsetbias'])
    parser.add_argument("--key_in_prompt", type=str, default=None)
    parser.add_argument("--GT_label", type=str, default='better_answer')
    parser.add_argument("--max_generation_length", type=int, default=512)

    # parser.add_argument("--task", type=str, default='wmt', choices=['fol', 'wmt', 'hha'])
    # parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/WMT_newstest2017-2020/newstest2017-2020_en-de.json")
    # parser.add_argument("--eval_data_path", type=str, default="/root/workspace/FOL-RL/datasets/WMT22_competition/WMT22_competition_en-de.json")
    # parser.add_argument("--reward_type", type=str, default=None, choices=['fol', 'offsetbias'])
    # parser.add_argument("--key_in_prompt", type=str, default='src')
    # parser.add_argument("--GT_label", type=str, default='ref')
    # parser.add_argument("--max_generation_length", type=int, default=1024)

    # parser.add_argument("--task", type=str, default='translate_to_fol', choices=['translate_to_fol', 'wmt', 'hha'])
    # parser.add_argument("--train_data_path", type=str, default="/root/workspace/FOL-RL/datasets/FOLIO/folio_with_steps.json")
    # parser.add_argument("--eval_data_path", type=str, default=None)
    # parser.add_argument("--reward_type", type=str, default=None, choices=['fol', 'offsetbias'])
    # parser.add_argument("--key_in_prompt", type=str, default='NL')
    # parser.add_argument("--GT_label", type=str, default='FOL')
    # parser.add_argument("--max_generation_length", type=int, default=512)

    parser.add_argument("--eval_batch_size", type=int, default=16)

    args = parser.parse_args()

    train(args)
    save_config(args)

from datasets import load_dataset
# import bert_score
import argparse
import numpy as np
import pandas as pd
import transformers
import copy
import logging
import torch
from peft import LoraConfig, prepare_model_for_int8_training
from tqdm import tqdm
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_metric
# from evaluate import load


# data config
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset

metric = load_metric("accuracy")

# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "<pad>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"
# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context.\n"
#         "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
#         "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task.\n"
#         "아래는 작업을 설명하는 명령어입니다.\n\n"
#         "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
#     ),
# }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'])
    parser.add_argument('--train_path', type=str, default='./SFT_train.json')
    parser.add_argument('--test_path', type=str, default=True)
    parser.add_argument('--reward_data_path', type=str, default=True)
    parser.add_argument('--model_name', type=str, choices=['gpt2', 'gpt3'])
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    # parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_ratio', type=float, default=0.5)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='/app/outputs')
    parser.add_argument('--overwrite_output_dir', type=bool, default=True)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--seq_length", type=int, default=1024)

    args = parser.parse_args()

    if args.model_name == 'gpt3':
        print(f'gpt3 use start')
        args.model_name = 'EleutherAI/polyglot-ko-1.3b'
    if args.model_name == 'gpt2':
        print(f'gpt2 use start')
        args.model_name = 'skt/kogpt2-base-v2'

    args.log_dir = os.path.join(args.output_dir, "logs")
    print(f'{args}')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)

def prepare_sample_text(example):
    text = f"Question: {example['cleaned_question']}\n\nAnswer: {example['clenaed_answer']}"
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(f'logits : {logits}')
    print(f'labels : {labels}')
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def create_datasets(tokenizer, args):
    dataset = load_dataset(
        'csv',
        data_files=args.train_path,
        split='train',
        use_auth_token=True,
    )
    dataset = dataset.train_test_split(test_size=0.15, seed=args.seed)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")

    # chars_per_token = chars_token_ratio(train_data, tokenizer)
    # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    #
    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     formatting_func=prepare_sample_text,
    #     infinite=True,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    # )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     valid_data,
    #     formatting_func=prepare_sample_text,
    #     infinite=False,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    # )
    return train_dataset, valid_dataset

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # cpu_state_dict = {key: value.cpu() for key, value in list(state_dict.items())}
        cpu_state_dict = {key: value for key, value in list(state_dict.items())}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def train(args):
    if args.model_name == 'EleutherAI/polyglot-ko-1.3b':
        print(f'####### gpt3 selected')
        print(f'####### load tokenizer')
        #['<|endoftext|>', '<|sep|>', '<|acc|>', '<|tel|>', '<|rrn|>']
        #[2, 3, 30000, 30001, 30002]
        #
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=1024,
        )

        print(f'####### create datasets')
        train_dataset, eval_dataset = create_datasets(tokenizer, args)

        print(f'####### set peft model')
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map={"": Accelerator().process_index},
            use_cache=False
        )

        model = prepare_model_for_int8_training(model)

    elif args.model_name == 'skt/kogpt2-base-v2':
        print(f'######## gpt2 loaded')
        # tokenizer = fn_tokenizer(args)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    print(f'######## model train set')
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        # max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=args.log_dir,  # directory for storing logs
        logging_steps=args.logging_steps, # same as eval_steps
        save_steps=args.save_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        # prediction_loss_only=True,
        save_strategy='steps',
        evaluation_strategy='steps',
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed
    )

    print(f'######## set SFTTrainer')
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        compute_metrics=compute_metrics,
        max_seq_length=1024,
        packing=False
    )

    print(f'######## model train start')
    trainer.train()
    # trainer.save_model()

    print(f'model save start')
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)
    # trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    print(f'model save end')

# def inference(args):
#     # tokenizer = fn_tokenizer(args)
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         args.model_name,
#         model_max_length=1024,
#     )
#     df = pd.read_csv(args.reward_data_path)
#     prompts = df['prompt'].to_list()
#     list_prompt = [PROMPT_DICT['prompt_no_input'].format_map({'prompt': prompt}) for prompt in prompts]
#     generator = pipeline('text-generation', model='bradmin/ployglot1.3', tokenizer=tokenizer, device=0)
#     generation_kwargs = dict(
#         repetition_penalty=2.0,
#         no_repeat_ngram_size=3,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         # min_length=200,
#         max_length=512,
#         do_sample=True,
#         top_k=0,
#         top_p=1.0
#     )
#     sft_answer = []
#     for i, instruction in enumerate(tqdm(list_prompt)):
#         result = generator([instruction], **generation_kwargs)
#         response = result[0][0]['generated_text'].split('Response(응답):')[1]
#         print(f'{i} , {response}')
#         sft_answer.append(response)
#
#     df['sft_answer'] = sft_answer
#     bertscore = load('bertscore')
#     bert_score_sft = bert_score.score(df['sft_answer'].astype(str).to_list(),
#                                       df['completion'].to_list(),
#                                       batch_size=32,
#                                       rescale_with_baseline=False,
#                                       # lang='others',
#                                       model_type='roberta-large',
#                                       idf=True,
#                                       verbose=True,
#                                       device=0)
#     df['bert_score_sft_P'] = bert_score_sft[0]
#     df['bert_score_sft_R'] = bert_score_sft[1]
#     df['bert_score_sft_F1'] = bert_score_sft[2]
#
#     bert_score_chatgpt = bert_score.score(df['chatgpt_answer'].astype(str).to_list(),
#                                       df['completion'].to_list(),
#                                       batch_size=32,
#                                       rescale_with_baseline=False,
#                                       # lang='others',
#                                       model_type='roberta-large',
#                                       idf=True,
#                                       verbose=True,
#                                       device=0)
#     df['bert_score_chatgpt_P'] = bert_score_chatgpt[0]
#     df['bert_score_chatgpt_R'] = bert_score_chatgpt[1]
#     df['bert_score_chatgpt_F1'] = bert_score_chatgpt[2]
#
#     df.to_csv(os.path.join(args.output_dir, 'result.csv'), index=False)

if __name__ == '__main__':
    main()
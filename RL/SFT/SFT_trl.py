from datasets import load_dataset
# import bert_score
import argparse
import numpy as np
import transformers
import pandas as pd
from peft import LoraConfig, prepare_model_for_int8_training
from tqdm import tqdm
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from evaluate import load
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

metric = load_metric("accuracy")
os.environ["WANDB_API_KEY"] = "8175d3b6ac05eaa98cbcbbd69dcbc55f7b4f0a6e"
tokenizer = None
DEFAULT_PAD_TOKEN = "<pad>"

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
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_ratio', type=float, default=0.5)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=100)
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(f'logits : {logits}')
    print(f'labels : {labels}')
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def create_datasets(args):
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
    return train_dataset, valid_dataset

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
    for i in range(len(example['cleaned_question'])):
        text = f"###Question: {example['cleaned_question'][i]}\n\n###Answer: {example['cleaned_answer'][i]}"
        output_texts.append(text)
    return output_texts
def prompt_text(example):
    text = f"###Question: {example['cleaned_question']}\n\n###Answer: "
    return text

def train(args):
    if args.model_name == 'EleutherAI/polyglot-ko-1.3b':
        print(f'####### gpt3 selected')
        print(f'####### load tokenizer')
        #['<|endoftext|>', '<|sep|>', '<|acc|>', '<|tel|>', '<|rrn|>']
        #[2, 3, 30000, 30001, 30002]
        #
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=1024,
            pad_token=DEFAULT_PAD_TOKEN
        )

        print(f'####### create datasets')
        train_dataset, eval_dataset = create_datasets(args)

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
            # load_in_8bit=True,
            # device_map={"": Accelerator().process_index},
            # use_cache=False
        )

        model = prepare_model_for_int8_training(model)

        response_template = "###Answer:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

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
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=args.log_dir,  # directory for storing logs
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        prediction_loss_only=True,
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
        data_collator=collator,
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
    print(f'model save end')

def inference(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=1024,
    )
    df = pd.read_csv('C:\\funnywork\\pycharm\\pythonProject\\Doctor-QA-with-RL\\RL\\data\\original\\PPO.csv')
    list_prompt = [prompt_text(df.iloc[i]) for i in range(len(df))]
    print(f'{args.model_name} loaded for inference')
    # generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device='cpu')
    generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device=0)
    generation_kwargs = dict(
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # min_length=200,
        max_length=512,
        do_sample=True,
        top_k=0,
        top_p=1.0
    )
    sft_answer = []
    for i, instruction in enumerate(tqdm(list_prompt)):
        result = generator([instruction], **generation_kwargs)
        response = result[0][0]['generated_text'].split('Answer:')[1]
        print(f'{i} , {response}')
        sft_answer.append(response)

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
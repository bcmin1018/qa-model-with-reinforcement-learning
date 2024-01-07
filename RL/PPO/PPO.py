# from evaluate import load
# import bert_score
import bert_score
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import pipeline, AutoTokenizer
import argparse
import pandas as pd
from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from trl import PPOTrainer
from trl import PPOConfig
from tqdm import tqdm
import os
import torch

tokenizer = None
DEFAULT_PAD_TOKEN = "<|pad|>"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'])
    parser.add_argument('--model_name', type=str, default='bradmin/sft_trl')
    parser.add_argument('--reward_model_name', type=str)
    parser.add_argument('--reward_tokenizer', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument('--train_path', type=str, default=True)
    parser.add_argument('--eval_path', type=str, default=True)
    parser.add_argument('--data_path', type=str, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mini_batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1.41e-5)
    parser.add_argument('--target_kl', type=float, default=0.1)
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='/app/outputs')
    parser.add_argument('--adap_kl_ctrl', type=bool, default=False)
    parser.add_argument('--init_kl_coef', type=float, default=0.2)
    parser.add_argument('--output_min_length', type=int, default=100)
    parser.add_argument('--output_max_length', type=int, default=200)
    parser.add_argument('--kl_penalty', type=str, default="kl")
    parser.add_argument('--reward_baseline', type=float, default=0.0)
    parser.add_argument('--tracker_project_name', type=str, default='ppo')
    parser.add_argument('--use_score_scaling', type=bool, default=False)
    parser.add_argument('--use_score_norm', type=bool, default=False)
    parser.add_argument('--score_clip', type=float, default=0.0)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.output_dir, "logs")
    print(f'###################설정 값 {args}########################')

    if args.mode == 'train':
        train(args)
    else:
        print(f'################eval mode#########################')
        eval(args)

def load_tokenizer(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        model_max_length=args.seq_length,
        pad_token=DEFAULT_PAD_TOKEN
    )
    return tokenizer

def preprocess_function(examples):
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    for question in examples['cleaned_question']:
        query = f"###Question: {question}\n\n###Answer: "
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])

    return new_examples

def prompt_text(example):
    text = f"###Question: {example}\n\n###Answer: "
    return text

def load_data(args):
    dataset = load_dataset(
        'csv',
        data_files=args.train_path,
        split='train',
        use_auth_token=True,
        cache_dir='/tmp/cache'
    )
    dataset = dataset.map(
        preprocess_function,
        batched=True
    )
    dataset.set_format('pt')
    return dataset

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def train(args):
    tokenizer = load_tokenizer(args)
    dataset = load_data(args)
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    active_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map={"": current_device},
        use_cache=False,
        peft_config=lora_config,
    )
    ref_model = create_reference_model(active_model)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        early_stopping=True,
        log_with='wandb',
        target_kl=args.target_kl,
        seed=2023,
        adap_kl_ctrl=False,
        init_kl_coef=args.init_kl_coef,
        remove_unused_columns=False,
        kl_penalty=args.kl_penalty,
        tracker_project_name=args.tracker_project_name,
        score_clip=args.score_clip
    )

    optimizer = torch.optim.AdamW(active_model.parameters(), lr=config.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000000006)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.000004, epochs=10, steps_per_epoch=10, pct_start=0.1, anneal_strategy='linear')

    ppo_trainer = PPOTrainer(
        config,
        active_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = ppo_trainer.accelerator.device

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_tokenizer)
    reward_model = pipeline("text-classification", args.reward_model_name, tokenizer=reward_tokenizer, device=device, return_token_type_ids=False)
    generation_kwargs = dict(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=0.0,
        top_p=1.0,
        min_length=-1,
        max_new_tokens=200,
        min_new_tokens=10
    )

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query,
                                            return_prompt=False, #true로 하면 step에서 q + r할때 프롬프트가 이중으로 들어간다.
                                            **generation_kwargs
                                            )
            response_tensors.append(response.squeeze())
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        texts = [q.strip() + " " + r.strip() for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts, padding=True, truncation=True, max_length=512)
        rewards = [torch.tensor(output["score"] - args.reward_baseline) for output in pipe_outputs]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(args.output_dir + f"/step_{epoch}")

def eval(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        model_max_length=args.seq_length,
    )
    df = pd.read_csv(args.eval_path)
    print(f'총 데이터 수 {len(df)}')
    list_prompt = df['cleaned_question'].apply(prompt_text)
    generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device=0)
    generation_kwargs = dict(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=0.0,
        top_p=1.0,
        min_length=-1,
        max_new_tokens=200,
        min_new_tokens=10
    )
    ppo_answer = []
    for i, instruction in enumerate(tqdm(list_prompt)):
        result = generator([instruction], **generation_kwargs)
        response = result[0][0]['generated_text'].split('Answer: ')[1]
        print(f'{i} ###Question: {instruction}\n\n ###Answer: {response}')
        ppo_answer.append(response)

    df[f'{args.model_name}_answer'] = ppo_answer
    df.to_csv(os.path.join(args.output_dir, f'inference_tmp.csv'), index=False)

    bert_score_sft = bert_score.score(df[f'{args.model_name}_answer'].astype(str).to_list(),
                                      df['cleaned_answer'].to_list(),
                                      batch_size=32,
                                      rescale_with_baseline=False,
                                      lang='others',
                                      model_type='roberta-large',
                                      idf=True,
                                      verbose=True,
                                      device=0)
    df[f'bert_score_{args.model_name}_P'] = bert_score_sft[0]
    df[f'bert_score_{args.model_name}_R'] = bert_score_sft[1]
    df[f'bert_score_{args.model_name}_F1'] = bert_score_sft[2]
    df.to_csv(os.path.join(args.output_dir, f'result_inference.csv'), index=False)

    average_f1 = df[f'bert_score_{args.model_name}_F1'].mean()
    print(f'average_f1 is {average_f1}')

if __name__ == '__main__':
    main()
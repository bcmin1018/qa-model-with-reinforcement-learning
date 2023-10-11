from evaluate import load
import bert_score
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import pipeline, AutoTokenizer
import argparse
import pandas as pd
import os
from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from trl import PPOTrainer
from trl import PPOConfig
from trl.core import LengthSampler
from tqdm import tqdm
import os
import wandb
# from random import choices
import transformers
import torch
# import numpy as np

os.environ["WANDB_API_KEY"] = "8175d3b6ac05eaa98cbcbbd69dcbc55f7b4f0a6e"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
access_token="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"

tokenizer = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'])
    parser.add_argument('--train_path', type=str, default=True)
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

    args = parser.parse_args()
    args.log_dir = os.path.join(args.output_dir, "logs")
    print(f'###################설정 값 {args}########################')

    if args.mode == 'train':
        train(args)
    else:
        print(f'inference mode')
        inference(args)

def load_tokenizer(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/polyglot-ko-1.3b',
        model_max_length=1024,
    )
    return tokenizer

def preprocess_function(examples):
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    for question in examples['cleaned_question']:
        query = f"### Question: {question}\n\n### Answer: "
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])

    return new_examples

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
        'bradmin/sft',
        load_in_8bit=True,
        device_map={"": current_device},
        peft_config=lora_config,
    )
    ref_model = create_reference_model(active_model)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        early_stopping=args.early_stopping,
        log_with='wandb',
        # tracker_project_name=""
        # log_with='tensorboard',
        # project_kwargs={"logging_dir": args.log_dir},
        target_kl=args.target_kl,
        seed=2023,
        adap_kl_ctrl=args.adap_kl_ctrl,
        init_kl_coef=args.init_kl_coef,
        remove_unused_columns=False,
        kl_penalty=args.kl_penalty
    )


    # optimizer = torch.optim.SGD(active_model.parameters(), lr=config.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    ppo_trainer = PPOTrainer(
        config,
        active_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        # optimizer=optimizer,
        # lr_scheduler=lr_scheduler
    )

    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = ppo_trainer.accelerator.device

    reward_model = pipeline("text-classification", "bradmin/reward", device=device)
    generation_kwargs = dict(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        # min_length=200,
        max_length=512,
        do_sample=True,
        top_k=0.0,
        top_p=1.0, # 상위 20% 토큰만 사용해서 고려
        min_length=-1
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
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # batch['response'] = [tokenizer.decode(torch.cat([query.squeeze(), response.squeeze()]), skip_special_tokens=True)
        #                      for query, response in zip(query_tensors, response_tensors)]
        # pipe_outputs = reward_model(batch['response'], padding=True, truncation=True, max_length=512)
        pipe_outputs = reward_model(texts, padding=True, truncation=True, max_length=512)
        # try:
        #     for i, response in enumerate(batch['response']):
        #         r = response.split('Response(응답):')[1]
        #         print(f'응답{i + 1}: {r}\n')
        # except IndexError:
        #     print(f'오류 발생 prompt: {i}  {r}')
        # print(rewards)

        rewards = [torch.tensor(output["score"] - args.reward_baseline) for output in pipe_outputs]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            # active_model.save_pretrained(args.output_dir + f"/step_{epoch}")
            ppo_trainer.save_pretrained(args.output_dir + f"/step_{epoch}")
            # ppo_trainer.push_to_hub(
            #     'bradmin/ppo_adapter',
            #     token=access_token
                # use_temp_dir=True,
                # use_auth_token=access_token
            # )


def inference(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/polyglot-ko-1.3b',
        model_max_length=1024,
    )
    df = pd.read_csv(args.train_path)
    prompts = df['prompt'].to_list()
    list_prompt = [PROMPT_DICT['prompt_no_input'].format_map({'prompt': prompt}) for prompt in prompts]
    generator = pipeline('text-generation', model="bradmin/ppo_model", tokenizer=tokenizer, device=0)
    generation_kwargs = dict(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        min_length=-1,
        max_length=512,
        do_sample=True,
        top_k=0.0,
        top_p=1.0
    )
    ppo_answer = []
    for i, instruction in enumerate(tqdm(list_prompt)):
        result = generator([instruction], **generation_kwargs)
        response = result[0][0]['generated_text'].split('Response(응답):')[1]
        print(f'{i} , {response}')
        ppo_answer.append(response)

    df['ppo_answer'] = ppo_answer
    bertscore = load('bertscore')
    bert_score_sft = bert_score.score(df['ppo_answer'].astype(str).to_list(),
                                      df['completion'].to_list(),
                                      batch_size=32,
                                      rescale_with_baseline=False,
                                      # lang='others',
                                      model_type='roberta-large',
                                      idf=True,
                                      verbose=True,
                                      device=0)
    df['bert_score_ppo_P'] = bert_score_sft[0]
    df['bert_score_ppo_R'] = bert_score_sft[1]
    df['bert_score_ppo_F1'] = bert_score_sft[2]

    df.to_csv(os.path.join(args.output_dir, 'result.csv'), index=False)

if __name__ == '__main__':
    main()
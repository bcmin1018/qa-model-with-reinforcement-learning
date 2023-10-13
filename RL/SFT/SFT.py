# 참고 영상 : https://www.youtube.com/watch?v=Iq8erq62s8c
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
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_metric
import bert_score
from evaluate import load
from trl import SFTTrainer

metric = load_metric("accuracy")
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    # "prompt_no_input": (
    #     "Below is an instruction that describes a task.\n"
    #     "아래는 작업을 설명하는 명령어입니다.\n\n"
    #     "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
    #     "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    # ),
    "prompt_no_input": (
        "###Question: {prompt}\n\n###Answer: "
    ),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'])
    parser.add_argument('--train_path', type=str, default='./SFT_train.json')
    parser.add_argument('--test_path', type=str, default='./test.csv')
    parser.add_argument('--inference_model', type=str, default='EleutherAI/polyglot-ko-1.3b')
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
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='/app/outputs')
    parser.add_argument('--overwrite_output_dir', type=bool, default=True)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')


    args = parser.parse_args()
    # args.output_dir = '/app/outputs'
    if args.model_name == 'gpt3':
        # args.model_name = 'skt/ko-gpt-trinity-1.2B-v0.5'
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

class SFT_dataset(Dataset):
    '''SFT dataset by wygo'''

    def __init__(self, data, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")

        ## format
        pattern_instruction = 'prompt'  # instruction
        pattern_input = 'input'  # 내 데이터엔 input이 없다
        pattern_output = 'completion'  # output

        ############################################################
        ## load dataset
        # 내 데이터셋엔 input이 없다
        # with open(os.path.join(data_path_1_SFT, 'SFT_train.json'), "r", encoding='utf-8-sig') as json_file:
        # with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
        #     list_data_dict = json.load(json_file)
        #     if verbose:
        #         print('## data check ##')
        #         print((list_data_dict[0]))

        # csv 부분
        list_data_dict = []
        for _, row in data.iterrows():
            row_dict = {}
            row_dict['prompt'] = row['cleaned_question']
            row_dict['completion'] = row['cleaned_answer']
            list_data_dict.append(row_dict)


        ############################################################
        ## 데이터셋 만들기, source와 target
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]  # 템플릿 가져오기

        # 입력
        sources = []
        for example in list_data_dict:
            if example.get(pattern_input, "") != "":
                tmp = prompt_input.format_map(example)
            else:
                tmp = prompt_no_input.format_map(example)
            sources.append(tmp)

        # 출력
        targets = []
        for example in list_data_dict:
            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        ############################################################
        # data_dict = preprocess(sources, targets, tokenizer)  # https://github.com/Beomi/KoAlpaca/blob/04704348d58b8b1c2e2638d6437a04b4e8ba1823/train.py#L124
        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized
        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source만
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target

        ## 입력은 source, 출력은 source+target 이지만 학습은 target 부분만
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX  # source 부분은 -100으로 채운다

        data_dict = dict(input_ids=input_ids, labels=labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d" % (len(self.labels)))

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # cpu_state_dict = {key: value.cpu() for key, value in list(state_dict.items())}
        cpu_state_dict = {key: value for key, value in list(state_dict.items())}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def fn_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        bos_token=DEFAULT_BOS_TOKEN,
        eos_token=DEFAULT_EOS_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
        pad_token=DEFAULT_PAD_TOKEN,
        padding_side="right",
        model_max_length=1024
    )
    return tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(f'logits : {logits}')
    print(f'labels : {labels}')
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train(args):
    if args.model_name == 'EleutherAI/polyglot-ko-1.3b':
        print(f'####### gpt3 loaded')
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=1024,
        )
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
        tokenizer = fn_tokenizer(args)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    df = pd.read_csv(args.train_path)
    train_data, eval_data = train_test_split(df, test_size=0.15, shuffle=True)
    train_dataset = SFT_dataset(train_data, tokenizer=tokenizer)
    eval_dataset = SFT_dataset(eval_data, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

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
        seed=2023
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        packing=True,
    )

    trainer.train()
    trainer.save_model()

    print(f'model save start')
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)
    print(f'model save end')

def generate_anwer(question):
    list_prompt = [PROMPT_DICT['prompt_no_input'].format_map({'prompt': question})]


def inference(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=1024,
    )
    df = pd.read_csv(args.test_path)
    prompts = df['cleaned_question'].to_list()
    list_prompt = [PROMPT_DICT['prompt_no_input'].format_map({'prompt': prompt}) for prompt in prompts]
    generator = pipeline('text-generation', model=args.inference_model, tokenizer=tokenizer, device=0)
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
    answer = []
    for i, instruction in enumerate(tqdm(list_prompt)):
        result = generator([instruction], **generation_kwargs)
        response = result[0][0]['generated_text'].split('###Answer: ')[1]
        print(f'{i} , {response}')
        answer.append(response)

    df[f'{args.inference_model}_answer'] = answer
    # bertscore = load('bertscore')
    bert_score_sft = bert_score.score(df[f'{args.inference_model}_answer'].astype(str).to_list(),
                                      df['cleaned_answer'].to_list(),
                                      batch_size=32,
                                      rescale_with_baseline=False,
                                      # lang='others',
                                      model_type='roberta-large',
                                      idf=True,
                                      verbose=True,
                                      device=0)
    df[f'{args.inference_model}_bert_score_P'] = bert_score_sft[0]
    df[f'{args.inference_model}_bert_score_R'] = bert_score_sft[1]
    df[f'{args.inference_model}_bert_score_F1'] = bert_score_sft[2]

    df.to_csv(os.path.join(args.output_dir, 'result.csv'), index=False)


if __name__ == '__main__':
    main()




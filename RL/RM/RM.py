from dataclasses import dataclass, field
from typing import Optional
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import RewardTrainer

DEFAULT_PAD_TOKEN = "<|pad|>"

@dataclass
class ScriptArguments:
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    num_labels: Optional[int] = field(default=1)
    train_path: Optional[str] = field(
        default=None,
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    max_steps: Optional[int] = field(
        default=-1,
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    output_dir: Optional[str] = field(
        default="/app/outputs",
    )
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    save_steps: Optional[int] = field(default=100)
    hub_model_id: Optional[str] = field(default="bradmin/reward-gpt")

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(f'{script_args}')
accuracy = evaluate.load("accuracy")

if script_args.model_name == 'gpt3':
    print(f'gpt3 use start')
    script_args.model_name = 'EleutherAI/polyglot-ko-1.3b'
if script_args.model_name == 'gpt2':
    print(f'gpt2 use start')
    script_args.model_name = 'skt/kogpt2-base-v2'

tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
if tokenizer_name == 'skt/kogpt2-base-v2':
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        bos_token="<pad>",
        eos_token="</s>",
        unk_token="</s>",
        pad_token="</s>",
        padding_side="right",
        model_max_length=script_args.max_length
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length=script_args.max_length,
        pad_token=DEFAULT_PAD_TOKEN
    )

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for question, response_j, response_k in zip(examples['question'], examples['response_j'], examples['response_k']):
        tokenized_j = tokenizer("###Question: " + question + "\n\n###Answer: " + response_j + tokenizer.eos_token, truncation=True)
        tokenized_k = tokenizer("###Question: " + question + "\n\n###Answer: " + response_k + tokenizer.eos_token, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def create_datasets(args):
    dataset = load_dataset(
        'csv',
        data_files=args.train_path,
        split='train',
        use_auth_token=True,
        cache_dir='/tmp/cache'
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=2023, shuffle=False)
    data_splits = dataset.map(preprocess_function, batched=True)

    train_dataset = data_splits["train"]
    valid_dataset = data_splits["test"]

    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=script_args.num_labels,
)

model.config.pad_token_id = tokenizer.eos_token_id
train_dataset, eval_dataset = create_datasets(script_args)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    eval_steps=script_args.eval_steps,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    adam_beta2=0.95,
    seed=2023
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    max_length=script_args.max_length
)

trainer.train()
trainer.save_model(script_args.output_dir)
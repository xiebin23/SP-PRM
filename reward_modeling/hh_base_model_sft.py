import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "weepcat/hh-rlhf"
split = "train"
streaming = True
size_valid_set = 4000
shuffle_buffer = 5000
seq_length = 1024
num_workers = 4
use_bnb = True
lora_alpha = 16
lora_dropout = 0.05
lora_r = 8
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
set_seed(42)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = ""
    for qa in example['chosen']:
        if qa['role'] == 'user':
            text += f"Human: {qa['content']}\n\n"
        elif qa['role'] == 'assistant':
            text += f"Assistant: {qa['content']}\n\n"
        else:
            text += f"System: {qa['content']}\n\n"
    return text


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer, args, seed=None):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = None
if use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # device_map={"": Accelerator().local_process_index},
    cache_dir="/root/autodl-tmp/cache"
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/autodl-tmp/cache")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
training_args = TrainingArguments(
    output_dir="./sft",
    max_steps=500,
    logging_steps=10,
    save_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    group_by_length=False,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.05,
    optim="paged_adamw_32bit",
    bf16=True,
)


def create_datasets(tokenizer, seed=None):
    dataset = load_dataset(
        dataset_name,
        num_proc=num_workers,
        cache_dir="/root/autodl-tmp/cache"
    )

    train_data = dataset["train"]
    test_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(test_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    test_dataset = ConstantLengthDataset(
        tokenizer,
        test_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset, test_dataset


train_dataset, test_dataset = create_datasets(tokenizer, seed=training_args.seed)
# %%
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    # max_seq_length=None,
    formatting_func=prepare_sample_text,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()
# %%
trainer.save_model(training_args.output_dir)
output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
# %%
# Free memory for merging weights
# del base_model
if is_torch_xpu_available():
    torch.xpu.empty_cache()
elif is_torch_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16,
                                                 local_file_only=True)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

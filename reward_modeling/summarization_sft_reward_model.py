import os
import argparse
import torch
from datasets import load_dataset
from trainer import RewardTrainer, RewardDataCollatorWithPadding, compute_metrics
from utils.util import wrap_model_name, get_logger
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from utils.model_factory import ModelFactory
from utils.tokenizer_factory import TokenizerFactory
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
parser = argparse.ArgumentParser(description="")
parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name")
parser.add_argument("--dataset_name", type=str, required=True, help="weepcat/summarization")
parser.add_argument('--learning_rate', type=float, default=1e-6, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0., help="Weight decay")
parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size for training")
parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help="Batch size for evaluation")
parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
parser.add_argument('--eval_every_steps', type=int, default=10, help="Evaluate every n steps")
parser.add_argument('--save_every_steps', type=int, default=10, help="Save model every n steps")
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Number of gradient accumulation steps")
parser.add_argument('--gradient_checkpointing', type=bool, default=True, help="Use gradient checkpointing")
parser.add_argument('--deepspeed', type=str, default=None, help="Deepspeed configuration file")
parser.add_argument('--bf16', type=bool, default=True, help="Use bfloat16")
parser.add_argument('--optim', type=str, default="adamw_torch", help="Optimizer")
parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler")
parser.add_argument('--cache_dir', type=str, default='/root/autodl-tmp/cache/', help="")
parser.add_argument('--local_rank', type=int, default=-1, help="")
parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps")
parser.add_argument('--logging_dir', type=str, default='./logs/train/', help="")
parser.add_argument('--report_to', type=str, default="wandb", help="Report to Hugging Face Hub")
args = parser.parse_args()
logger = get_logger()


if __name__ == "__main__":
    logger.info(args)
    logger.info(f"Loading model {args.model_name}...")
    logger.info(f"Loading tokenizer {args.model_name}...")
    model = ModelFactory.load_reward_model(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = TokenizerFactory.load_tokenizer(
        args.model_name,
        cache_dir=args.cache_dir
    )
    train_ds = load_dataset(args.dataset_name, cache_dir=args.cache_dir, split="train_prefs").shuffle(seed=42)
    test_ds = load_dataset(args.dataset_name, cache_dir=args.cache_dir, split="test_prefs")
    logger.info("Training set: %d, Test set: %d" % (len(train_ds), len(test_ds)))
    data_collator = RewardDataCollatorWithPadding(
        tokenizer=tokenizer,
    )
    repo_id, model_name = args.model_name.split("/")
    output_dir = f"summarization_sft_{model_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_every_steps,
        save_strategy="steps",
        save_steps=args.save_every_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=args.bf16,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_dir=str(os.path.join(args.logging_dir, args.model_name)),
        report_to=args.report_to,
        batch_eval_metrics=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    model.config.use_cache = not args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    print("Training finished!")

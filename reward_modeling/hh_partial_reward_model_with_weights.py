import os
import argparse
import torch
from datasets import load_dataset, Dataset
from trainer import RewardTrainerWithWeight, MaskWeightedRewardDataCollatorWithPadding, compute_metrics
from utils.util import get_logger, wrap_model_name
from transformers import TrainingArguments
from utils.model_factory import ModelFactory
from utils.tokenizer_factory import TokenizerFactory
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser = argparse.ArgumentParser(description="")
parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name")
parser.add_argument('--ref_model', type=str, required=True)
parser.add_argument("--train_dataset_name", type=str, required=True)
parser.add_argument("--partial_process_method", type=str, choices=['token_by_token', "random_length"], required=True)
# parser.add_argument("--eval_dataset_name", type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./models/', help="Directory to save evaluation results")
parser.add_argument('--learning_rate', type=float, default=1e-6, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0., help="Weight decay")
parser.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size for training")
parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help="Batch size for evaluation")
parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs")
parser.add_argument('--eval_every_steps', type=int, default=30, help="Evaluate every n steps")
parser.add_argument('--save_every_steps', type=int, default=30, help="Save model every n steps")
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Number of gradient accumulation steps")
parser.add_argument('--gradient_checkpointing', type=bool, default=True, help="Use gradient checkpointing")
parser.add_argument('--deepspeed', type=str, default=None, help="Deepspeed configuration file")
parser.add_argument('--bf16', type=bool, default=True, help="Use bfloat16")
parser.add_argument('--optim', type=str, default="adamw_torch", help="Optimizer")
parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler")
parser.add_argument('--cache_dir', type=str, default='/root/autodl-tmp/cache/', help="")
parser.add_argument('--local_files_only', type=bool, default=False, help="")
parser.add_argument('--local_rank', type=int, default=-1, help="")
parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps")
parser.add_argument('--logging_dir', type=str, default='./logs/train/', help="")
parser.add_argument('--shutdown', type=bool, default=False, help="Shutdown after training")
parser.add_argument('--report_to', type=str, default="wandb", help="Report to Hugging Face Hub")
args = parser.parse_args()


def filter_train_dataset(train_ds: Dataset, ref_model: str):
    idx = [i for i, weight in enumerate(train_ds[f"{wrap_model_name(ref_model)}_masked_weights"]) if weight > 0]
    return train_ds.select(idx)


if __name__ == "__main__":
    logger = get_logger()
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

    # 加载数据集
    # train_dataset_dir = os.path.join(args.cache_dir, "hh_pref_train_partial")
    # eval_dataset_dir = os.path.join(args.cache_dir, "hh_pref_eval_partial")
    # logger.info(f"Loading training dataset from {train_dataset_dir}...")
    # logger.info(f"Loading evaluation dataset from {eval_dataset_dir}...")
    train_ds = load_dataset(args.train_dataset_name, cache_dir=args.cache_dir, split="train").shuffle(seed=42)
    train_ds = filter_train_dataset(train_ds, args.ref_model)
    # eval_ds = load_dataset(args.eval_dataset_name, cache_dir=args.cache_dir, split="test")
    # logger.info("Training set: %d, Eval set: %d" % (len(train_ds), len(eval_ds)))
    data_collator = MaskWeightedRewardDataCollatorWithPadding(
        tokenizer=tokenizer,
        mask="mask",
        ref_model=args.ref_model,
    )

    repo_id, model_name = args.model_name.split("/")
    _, ref_model = args.ref_model.split("/")
    output_dir = f"{model_name}_{ref_model}_mask_partial_rm_{args.partial_process_method}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        # eval_strategy="steps",
        # eval_steps=args.eval_every_steps,
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
        # batch_eval_metrics=True,
        # push_to_hub=True,
        # dataloader_num_workers=24,
        # save_only_model=True,
        save_total_limit=3,
        # load_best_model_at_end=True
    )

    model.config.use_cache = not args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    trainer = RewardTrainerWithWeight(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        # eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    print("Training finished!")

import random
from datasets import load_dataset, Dataset, concatenate_datasets
import os
import numpy as np
import argparse
from tqdm import tqdm
from typing import Callable
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
np.random.seed(42)
random.seed(42)


def process_dataset(
        train_ds: Dataset,
        test_ds: Dataset,
        generate_partial_answer_method: Callable,
        train_size: int = 4096,
        test_size: int = 128
):
    idx = np.random.choice(len(train_ds), size=train_size, replace=False)
    raw_train_dataset = train_ds.select(idx)
    idx = np.random.choice(len(test_ds), size=test_size, replace=False)
    raw_test_dataset = test_ds.select(idx)
    transformed_rows_train = generate_partial_answer_method(raw_train_dataset)
    transformed_rows_eval = generate_partial_answer_method(raw_test_dataset)
    train_ds = Dataset.from_list(transformed_rows_train)
    test_ds = Dataset.from_list(transformed_rows_eval)
    return train_ds, test_ds, raw_train_dataset, raw_test_dataset


def generate_partial_answer_by_random_length(dataset):
    transformed_rows = []
    for row in tqdm(dataset):
        w_ans_list = row['chosen'][-1]['content'].split()
        l_ans_list = row['rejected'][-1]['content'].split()
        chosen_len = len(w_ans_list)
        rejected_len = len(l_ans_list)
        max_length = max(chosen_len, rejected_len)

        # 生成1-10的随机数num
        num = random.randint(1, 10)
        # 生成num个不重复的数，范围在 1, max_length之间
        random_list = random.sample(range(1, max_length), num) if max_length > num else range(1, max_length)
        prompt = row['chosen'][:-1]
        for i in random_list:
            prompt.append({"content": " ".join(w_ans_list[0:i + 1]), "role": "assistant"})
            partial_chosen = prompt.copy()
            prompt.pop()
            prompt.append({"content": " ".join(l_ans_list[0:i + 1]), "role": "assistant"})
            partial_rejected = prompt.copy()
            prompt.pop()
            transformed_rows.append({
                'chosen': partial_chosen,
                'rejected': partial_rejected,
            })
    return transformed_rows


def generate_partial_answer_token_by_token(dataset):
    transformed_rows = []
    for row in tqdm(dataset):
        w_ans_list = row['chosen'][-1]['content'].split()
        l_ans_list = row['rejected'][-1]['content'].split()
        chosen_len = len(w_ans_list)
        rejected_len = len(l_ans_list)
        max_length = max(chosen_len, rejected_len)

        prompt = row['chosen'][:-1]
        for i in range(0, max_length):
            prompt.append({"content": " ".join(w_ans_list[0:i + 1]), "role": "assistant"})
            partial_chosen = prompt.copy()
            prompt.pop()
            prompt.append({"content": " ".join(l_ans_list[0:i + 1]), "role": "assistant"})
            partial_rejected = prompt.copy()
            prompt.pop()
            transformed_rows.append({
                'chosen': partial_chosen,
                'rejected': partial_rejected,
            })
    return transformed_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="weepcat/summarization", required=True)
    parser.add_argument("--partial_process_method", type=str, choices=['token_by_token', "random_length"], required=True)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache/", help="")
    parser.add_argument("--train_size", type=int, default=4096)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--tag", type=int, default=1)
    args = parser.parse_args()
    datasets = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    train_ds = datasets['train_prefs']
    test_ds = datasets['test_prefs']
    if args.partial_process_method == "token_by_token":
        partial_process_method = generate_partial_answer_token_by_token
        train_size = args.train_size
        test_size = args.test_size
    else:
        partial_process_method = generate_partial_answer_by_random_length
        train_size = len(train_ds) // args.tag
        test_size = len(test_ds) // args.tag
    train_ds, test_ds, raw_train_ds, raw_test_ds = process_dataset(
        train_ds,
        test_ds,
        partial_process_method,
        train_size,
        test_size
    )
    if args.partial_process_method == "token_by_token":
        train_ds.push_to_hub(f"weepcat/summarization_partial_reward_model_{args.partial_process_method}", split="train")
        test_ds.push_to_hub(f"weepcat/summarization_partial_reward_model_{args.partial_process_method}", split="test")
    else:
        train_ds = concatenate_datasets([train_ds, raw_train_ds], axis=0)
        test_ds = concatenate_datasets([test_ds, raw_test_ds], axis=0)
        train_ds.push_to_hub(f"weepcat/summarization_partial_reward_model_{args.partial_process_method}-{args.tag}", split="train")
        test_ds.push_to_hub(f"weepcat/summarization_partial_reward_model_{args.partial_process_method}-{args.tag}", split="test")

    # train_ds.save_to_disk(os.path.join(cache_dir, "hh_partial_reward_model.sh"))
    # test_ds.save_to_disk(os.path.join(cache_dir, "hh_pref_test_partial"))
    print("Done")

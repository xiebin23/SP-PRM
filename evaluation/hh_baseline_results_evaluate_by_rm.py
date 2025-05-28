import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from baselines.scorer import RewardModelScorer
from baselines.config import RMScorerConfig
import torch
import argparse
from utils.util import process_prompt_from_str_to_list, wrap_model_name, process_prompt_from_list_to_str
from typing import List, Any
from dataclasses import dataclass
from utils.model_factory import ModelFactory
from utils.tokenizer_factory import TokenizerFactory
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class to store command line arguments"""
    rm: str
    llm_name: str
    guided_rm: str
    sample_size: int
    seed: int
    batch_size: int
    cache_dir: str
    methods: List[str]
    partial_process_method: str
    # device: str = "cpu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments() -> Config:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate model responses using reward model")
    parser.add_argument("--rm", type=str, required=True)
    parser.add_argument("--llm_name", type=str, required=True,
                        help="alignment-handbook/zephyr-7b-sft-full")
    parser.add_argument("--guided_rm", type=str, required=True,
                        help="weepcat/hh_sft_RM-Gemma-2B")
    parser.add_argument("--partial_process_method", type=str, choices=['token_by_token', "random_length"], required=True)
    parser.add_argument("--methods", nargs='+', help='method list')
    parser.add_argument("--cache_dir", type=str, default='/root/autodl-tmp/cache')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger.info(args)
    return Config(**vars(args))


def construct_conversation(row: pd.Series, methods: List[str]) -> pd.Series:
    """Process conversation format based on LLM type"""
    for method in methods:
        row[method] = [{"content": row[method],
                       "role": "assistant"}]
    return row


def construct_responses_from_methods(row: pd.Series, methods: List[str]) -> List[Any]:
    """Extract responses for each method from the row"""
    return [row[method] for method in methods]


def process_response_from_list_to_str(row: pd.Series, methods: List[str]) -> pd.Series:
    for method in methods:
        row[method] = row[method][-1]['content']
    return row


def filter_methods(candidate_methods: List, methods: List):
    filtered_methods = []
    for candidate_method in candidate_methods:
        for method in methods:
            if method in candidate_method:
                filtered_methods.append(candidate_method)
    return filtered_methods


def load_and_process_data(config: Config, csv_path: str = None, is_exists: bool = False) -> tuple[pd.DataFrame, List[str]]:
    """Load and process the JSON data into a DataFrame"""
    suffix = f"sample_{config.sample_size}_seed_{config.seed}"
    json_file_path = os.path.join(
        "./results/evaluate/",
        wrap_model_name(config.llm_name),
        wrap_model_name(config.guided_rm),
        config.partial_process_method,
        f"baseline_results_{suffix}.json"
    )
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {json_file_path}")

    prompts = list(data.keys())
    methods = list(data[prompts[0]].keys())
    results = [
        {'prompt': prompt, **{method: data[prompt][method]
                              for method in methods}}
        for prompt in prompts
    ]
    json_to_df = pd.DataFrame(data=results)
    if is_exists:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, json_to_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        df = json_to_df
    df['prompt'] = df['prompt'].map(process_prompt_from_str_to_list)
    filtered_methods = filter_methods(methods, config.methods)
    print(filtered_methods)
    df = df.apply(lambda row: construct_conversation(row, filtered_methods),
                  axis=1)
    return df, filtered_methods


def calculate_scores(df: pd.DataFrame, methods: List[str],
                     rm_scorer: RewardModelScorer) -> np.ndarray:
    """Calculate scores for each response using the reward model"""
    scores = np.zeros((len(df), len(methods)))

    for i in tqdm(range(len(df)), desc="Calculating scores"):
        prompt = df.iloc[i]['prompt']
        responses = construct_responses_from_methods(df.iloc[i], methods)
        with torch.no_grad():
            try:
                scores[i, :] = rm_scorer.get_batched_reward(prompt, responses).squeeze().detach().cpu().numpy()
            except Exception as e:
                print(f"An error occurred for prompt: {prompt}")
                print(f"Error: {str(e)}")
                scores[i, :] = np.nan
    return scores


def save_results(df: pd.DataFrame, scores: np.ndarray, methods: List[str],
                 config: Config, csv_file_path: str):
    """Save the results to a CSV file"""
    score_columns = [f"{wrap_model_name(config.rm)}_{method}_score" for method in methods]
    df_scores = pd.DataFrame(data=scores, columns=score_columns)
    df['prompt'] = df['prompt'].map(process_prompt_from_list_to_str)
    df = df.apply(lambda row: process_response_from_list_to_str(row, methods), axis=1)
    final_df = pd.concat([df, df_scores], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df.to_csv(csv_file_path, index=False)
    print(f"Results saved to: {csv_file_path}")


def main():
    """Main execution function"""
    # Set up error handling and logging
    try:
        # Parse arguments
        config = parse_arguments()
        # Load and process data
        suffix = f"sample_{config.sample_size}_seed_{config.seed}"
        csv_file_path = os.path.join(
            "./results/evaluate/",
            wrap_model_name(config.llm_name),
            wrap_model_name(config.guided_rm),
            config.partial_process_method,
            f"baseline_results_{suffix}.csv"
        )

        df, methods = load_and_process_data(config, csv_file_path, os.path.exists(csv_file_path))
        # Initialize reward model
        rm = ModelFactory.load_reward_model(
            name_or_path=config.rm,
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16
        )

        rm_tokenizer = TokenizerFactory.load_tokenizer(
            name_or_path=config.rm,
            cache_dir=config.cache_dir
        )

        score_config = RMScorerConfig(batch_size=config.batch_size)
        rm_scorer = RewardModelScorer(
            rm=rm,
            rm_tokenizer=rm_tokenizer,
            config=score_config
        )
        # Calculate scores
        scores = calculate_scores(df, methods, rm_scorer)
        # Save results
        save_results(df, scores, methods, config, csv_file_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

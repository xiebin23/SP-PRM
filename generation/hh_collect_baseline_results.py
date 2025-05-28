import torch
from utils.model_factory import ModelFactory
from utils.tokenizer_factory import TokenizerFactory
import os
from datasets import load_dataset
import numpy as np
from baselines.scorer import RewardModelScorer
from baselines.config.scorer_config import RMScorerConfig
from utils.eval_config import EvalConfig
import json
import argparse
from tqdm import tqdm
import logging
from utils.generator_factory import GeneratorFactory
from utils.util import apply_chat_template, wrap_model_name
from torch.multiprocessing import Queue, Process
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
import torch.multiprocessing as mp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)

@dataclass
class BatchResult:
    prompt_id: int
    wrap_prompt: str
    method: str
    output: str


def parse_arguments() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_partial", type=str, required=True)
    parser.add_argument("--rm_full", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True, help="alignment-handbook/zephyr-7b-sft-full")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--partial_process_method", type=str, choices=['token_by_token', "random_length"], required=True)
    parser.add_argument("--dataset_name", type=str, default="weepcat/hh-rlhf-eval")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--n_bon", type=int, default=16)
    parser.add_argument("--w_args", type=float, default=1.0)
    parser.add_argument("--k_args", type=int, default=10)
    parser.add_argument("--w_cbs", type=int, default=2)
    parser.add_argument("--k_cbs", type=int, default=2)
    parser.add_argument("--l_cbs", type=int, default=30)
    parser.add_argument("--rt_cards", type=float, default=8.)
    parser.add_argument("--k_cards", type=int, default=30)
    parser.add_argument("--et_cards", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--methods", nargs='+', help='method list')
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=list(range(torch.cuda.device_count())))
    args = parser.parse_args()
    logger.info(args)
    return EvalConfig(**vars(args))


def wrap_method_to_string(method: str, config: EvalConfig) -> str:
    if "args" in method:
        wrap_method_name = f"{method}_w_{config.w_args}_top_k_{config.k_args}"
    elif "best_of_n" in method:
        wrap_method_name = f"{method}_{config.n_bon}"
    elif "vanilla_top_k" in method:
        wrap_method_name = f"{method}_{config.top_k}"
    elif "vanilla_top_p" in method:
        wrap_method_name = f"{method}_{config.top_p}"
    elif "cbs" in method:
        wrap_method_name = f"{method}_w_{config.w_cbs}_k_{config.k_cbs}_l_{config.l_cbs}"
    elif "cards" in method:
        wrap_method_name = f"{method}_rt_{config.rt_cards}_top_k_{config.k_cards}_et_{config.et_cards}"
    else:
        wrap_method_name = method
    return wrap_method_name


class GPUWorker:

    def __init__(self, config: EvalConfig, gpu_id: int, methods: List[str], batch_size: int = 4):
        self.rm_partial_tokenizer = None
        self.rm_full_tokenizer = None
        self.config = config
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.methods = methods
        self.batch_size = batch_size
        self.llm = None
        self.llm_tokenizer = None
        self.rm_full = None
        self.rm_partial = None
        self.rm_tokenizer = None
        self.rm_full_scorer = None
        self.rm_partial_scorer = None
        self.generators = None

    def initialize(self):
        torch.cuda.set_device(self.gpu_id)
        self.llm = ModelFactory.load_llm(
            self.config.llm,
            self.config.cache_dir,
            torch_dtype=torch.float16,
        )

        self.llm_tokenizer = TokenizerFactory.load_tokenizer(
            self.config.llm,
            self.config.cache_dir
        )

        self.rm_full = ModelFactory.load_reward_model(
            self.config.rm_full,
            self.config.cache_dir,
            torch_dtype=torch.float16
        )

        self.rm_full_tokenizer = TokenizerFactory.load_tokenizer(
            self.config.rm_partial,
            self.config.cache_dir,
        )

        self.rm_partial = ModelFactory.load_reward_model(
            self.config.rm_partial,
            self.config.cache_dir,
            torch_dtype=torch.float16
        )

        self.rm_partial_tokenizer = TokenizerFactory.load_tokenizer(
            self.config.rm_partial,
            self.config.cache_dir
        )

        score_config = RMScorerConfig()
        self.rm_full_scorer = RewardModelScorer(
            rm=self.rm_full,
            rm_tokenizer=self.rm_full_tokenizer,
            config=score_config
        )
        self.rm_partial_scorer = RewardModelScorer(
            rm=self.rm_partial,
            rm_tokenizer=self.rm_partial_tokenizer,
            config=score_config
        )
        self.generators = GeneratorFactory.create_generators(
            self.config,
            self.llm,
            self.llm_tokenizer,
            self.rm_full_scorer,
            self.rm_partial_scorer
        )
        logger.info(f"Successfully initialized all models on GPU {self.gpu_id}")

    async def process_batch(self, method: str, batch_data: List[Tuple[int, Dict]]) -> List[BatchResult]:
        generator = self.generators[method]

        async def process_single(prompt_id: int, prompt: Dict) -> BatchResult:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                output = await loop.run_in_executor(
                    executor, generator.generate, prompt
                )
                return BatchResult(
                    prompt_id=prompt_id,
                    wrap_prompt=apply_chat_template(prompt),
                    method=wrap_method_to_string(method, self.config),
                    output=output,
                )

        tasks = [process_single(pid, p) for pid, p in batch_data]
        return await asyncio.gather(*tasks)

    async def process_methods_concurrently(self, batch_data: List[Tuple[int, Dict]]) -> List[BatchResult]:
        """并发处理多个方法"""
        tasks = [self.process_batch(method, batch_data) for method in self.methods]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def cleanup(self):
        """清理GPU资源"""
        del self.llm, self.rm_full, self.rm_partial
        del self.rm_full_scorer, self.rm_partial_scorer
        del self.generators
        torch.cuda.empty_cache()


class ParallelEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.method_to_gpu = self._assign_methods_to_gpus()
        self.batch_size = config.batch_size  # 可配置的批处理大小

    def _assign_methods_to_gpus(self) -> Dict[str, int]:
        method_to_gpu = {}
        available_gpus = self.config.gpu_ids
        method_complexity = {
            'vanilla_greedy': 1,
            'vanilla_top_k': 1,
            'vanilla_top_p': 1,
            'best_of_n_full': 4,
            'best_of_n_partial': 4,
            'args_full_greedy': 3,
            'args_full_stochastic': 3,
            'args_partial_greedy': 3,
            'args_partial_stochastic': 3,
            'cbs_full': 2,
            'cbs_partial': 2,
            "cards_full": 2,
            "cards_partial": 2
        }
        sorted_methods = sorted(
            self.config.methods,
            key=lambda x: method_complexity.get(x, 1),
            reverse=True
        )
        gpu_loads = {gpu_id: 0 for gpu_id in available_gpus}
        for method in sorted_methods:
            min_load_gpu = min(gpu_loads.items(), key=lambda x: x[1])[0]
            method_to_gpu[method] = min_load_gpu
            gpu_loads[min_load_gpu] += method_complexity.get(method, 1)
        return method_to_gpu

    @staticmethod
    async def _worker_process(gpu_id: int,
                              methods: List[str],
                              config: EvalConfig,
                              input_queue: Queue,
                              result_queue: Queue,
                              batch_size: int):
        worker = GPUWorker(config, gpu_id, methods, batch_size)
        worker.initialize()
        batch_buffer = []
        while True:
            try:
                data = input_queue.get_nowait()
                if data is None:
                    if batch_buffer:
                        results = await worker.process_methods_concurrently(batch_buffer)
                        for result in results:
                            result_queue.put(result)
                    break
                batch_buffer.append(data)
                if len(batch_buffer) >= batch_size:
                    results = await worker.process_methods_concurrently(batch_buffer)
                    for result in results:
                        result_queue.put(result)
                    batch_buffer = []
            except Exception as e:
                logger.error(f"Error in worker process {gpu_id}: {str(e)}")
                raise
        worker.cleanup()

    def evaluate(self):
        mp.set_start_method('spawn', force=True)
        output_dir = os.path.join(
            "./results/evaluate/",
            f"{wrap_model_name(self.config.llm)}/",
            f"{wrap_model_name(self.config.rm_full)}",
            self.config.partial_process_method
        )
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"sample_{self.config.sample_size}_seed_{self.config.seed}"
        output_file_path = os.path.join(output_dir, f"baseline_results_{suffix}.json")
        results = {}
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as f:
                results = json.load(f)
        eval_ds = self._load_dataset()
        input_queues = {gpu_id: Queue() for gpu_id in self.config.gpu_ids}
        result_queue = Queue()
        processes = []
        gpu_to_methods = {}
        for method, gpu_id in self.method_to_gpu.items():
            if gpu_id not in gpu_to_methods:
                gpu_to_methods[gpu_id] = []
            gpu_to_methods[gpu_id].append(method)

        for gpu_id, methods in gpu_to_methods.items():
            p = Process(
                target=self._run_async_worker,
                args=(gpu_id, methods, self.config, input_queues[gpu_id],
                      result_queue, self.batch_size)
            )
            p.start()
            processes.append(p)

        # 分发数据
        for i, example in enumerate(tqdm(eval_ds, desc="Distributing data")):
            for gpu_id in gpu_to_methods.keys():
                input_queues[gpu_id].put((i, example['prompt']))

        # 发送结束信号
        for queue in input_queues.values():
            queue.put(None)

        self._collect_results(results, result_queue, len(eval_ds) * len(self.config.methods), output_file_path)
        # 清理进程
        for p in processes:
            p.join()

    @staticmethod
    def _run_async_worker(*args):
        """运行异步工作进程的包装器"""
        asyncio.run(ParallelEvaluator._worker_process(*args))

    def _collect_results(self, results: dict, result_queue: Queue, total_results: int, output_file_path: str):
        """收集和保存结果"""
        collected = 0
        with tqdm(total=total_results, desc="Collecting results") as pbar:
            while collected < total_results:
                result = result_queue.get()
                if result.wrap_prompt not in results:
                    results[result.wrap_prompt] = {}
                results[result.wrap_prompt][result.method] = result.output
                collected += 1
                pbar.update(1)
                if collected % 10 == 0:
                    torch.cuda.empty_cache()
                    self._save_results(results, output_file_path)

        self._save_results(results, output_file_path)

    def _save_results(self, results: Dict, output_file_path):
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _load_dataset(self):
        eval_ds = load_dataset(
            self.config.dataset_name,
            split="eval",
            cache_dir=self.config.cache_dir
        )
        idxs = np.random.choice(len(eval_ds), size=self.config.sample_size)
        return eval_ds.select(idxs)


def main():
    try:
        config = parse_arguments()
        evaluator = ParallelEvaluator(config)
        evaluator.evaluate()
    except Exception as e:
        logger.error(f"Failed to evaluate: {str(e)}")
        raise


if __name__ == "__main__":
    main()

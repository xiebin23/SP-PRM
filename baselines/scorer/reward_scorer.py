from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, List, Dict, Union
import torch
from utils.util import apply_chat_template
from baselines.config.scorer_config import RMScorerConfig
from baselines.scorer.base_scorer import BaseScorer


class RewardModelScorer(BaseScorer):
    def __init__(
        self,
        rm: PreTrainedModel,
        rm_tokenizer: PreTrainedTokenizer,
        config: Optional[RMScorerConfig] = None
    ):
        """
        初始化reward model评分器

        Args:
            rm: Reward模型
            rm_tokenizer: Reward模型的tokenizer
        """
        super().__init__()
        self.rm = rm
        self.rm_tokenizer = rm_tokenizer
        self.config = config or RMScorerConfig()
        self.rm.eval()

    def _process_prompt_and_responses(self, prompt: Union[List[Dict], str], responses: List[List[Dict]]) -> List[str]:
        """处理问题文本，使用缓存避免重复处理"""
        if self.rm_tokenizer.chat_template is not None:
            if self.rm_tokenizer.bos_token:
                if isinstance(prompt, list):
                    inputs = [self.rm_tokenizer.apply_chat_template(
                        prompt + response,
                        tokenize=False,
                        add_generation_prompt=False
                    ).replace(self.rm_tokenizer.bos_token, "") for response in responses]
                else:
                    inputs = [prompt + self.rm_tokenizer.apply_chat_template(
                        response,
                        tokenize=False,
                        add_generation_prompt=False
                    ).replace(self.rm_tokenizer.bos_token, "") for response in responses]
            else:
                if isinstance(prompt, list):
                    inputs = [self.rm_tokenizer.apply_chat_template(
                        prompt + response,
                        tokenize=False,
                        add_generation_prompt=False
                    ) for response in responses]
                else:
                    inputs = [prompt + self.rm_tokenizer.apply_chat_template(
                        response,
                        tokenize=False,
                        add_generation_prompt=False
                    ) for response in responses]
        else:
            if isinstance(prompt, list):
                inputs = [apply_chat_template(prompt + response) for response in responses]
            else:
                inputs = [prompt + apply_chat_template(response) for response in responses]
        return inputs

    def _batch_tokenize(
            self,
            batch_inputs: List[str]
    ) -> dict:
        """批量tokenize输入文本"""
        inputs = self.rm_tokenizer(
            batch_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        return {k: v.to(self.rm.device) for k, v in inputs.items()}

    @torch.no_grad()
    def get_batched_reward(
            self,
            prompt: Union[List[Dict], str],
            responses: List[List[Dict]],
            return_tensors: bool = True
    ) -> Union[torch.Tensor, List[float]]:
        """
        获取批量回答的reward分数

        Args:
            prompt: 单个问题
            responses: 回答列表
            return_tensors: 是否返回tensor，False则返回Python列表

        Returns:
            reward分数的tensor或列表
        """
        try:
            model_inputs = self._process_prompt_and_responses(prompt, responses)
            all_scores = []
            for i in range(0, len(model_inputs), self.config.batch_size):
                batch_inputs = model_inputs[i:i + self.config.batch_size]
                inputs = self._batch_tokenize(batch_inputs)
                outputs = self.rm(**inputs)
                # scores = outputs.logits
                scores = getattr(outputs, "logits", outputs)
                all_scores.append(scores)
            torch.cuda.empty_cache()
            final_scores = torch.cat(all_scores)
            if return_tensors:
                return final_scores
            return final_scores.cpu().tolist()

        except Exception as e:
            print(f"Error in get_batched_reward: {str(e)}")
            raise

    def __call__(
            self,
            prompt: Union[List[Dict], str],
            responses: List[List[Dict]],
            return_tensors: bool = True
    ) -> Union[torch.Tensor, List[float]]:
        return self.get_batched_reward(prompt, responses, return_tensors)

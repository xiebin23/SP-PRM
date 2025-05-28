from baselines.generator.base_generator import BaseGenerator
from baselines.scorer.reward_scorer import RewardModelScorer
from typing import Union, List, Dict
import random
import math
from utils.util import entropy
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import os
from baselines.config import RMScorerConfig, CARDSGeneratorConfig
import torch
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    EosTokenCriteria,
    StoppingCriteriaList
)
from dataclasses import dataclass
os.environ['cache_dir'] = "/root/autodl-tmp/cache"
cache_dir = "/root/autodl-tmp/cache"


@dataclass
class CARDSGenerator(BaseGenerator):
    scorer: RewardModelScorer
    config: CARDSGeneratorConfig

    def _generate(self, prompt: Union[List[Dict], str], **model_kwargs):
        stopping_criteria = StoppingCriteriaList()
        if self.config.max_new_tokens is None:
            stopping_criteria.append(MaxLengthCriteria(max_length=self.config.max_length))
        if self.llm_tokenizer.eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=self.llm_tokenizer.eos_token_id))


        llm_prompt = self.process_prompt(prompt)
        inputs = self.llm_tokenizer(
            llm_prompt,
            return_tensors='pt'
        )

        best_candidate, best_reward = None, -1e34
        num_regeneration = 0
        input_ids = inputs['input_ids'].to(self.llm.device)
        if self.config.max_new_tokens:
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1) + self.config.max_new_tokens))

        prompt_len = input_ids.size(1)
        reward0 = self.scorer(
            prompt=prompt,
            responses=self.wrap_responses([""])
        )
        reward0 = (1 - self.config.alpha) * reward0.item() + self.config.alpha * self.config.reward_threshold

        def accept_check(reward, candidate):
            threshold = reward0 + (candidate.shape[1] - prompt_len) * (self.config.reward_threshold - reward0) / self.config.max_new_tokens
            threshold = min(threshold, self.config.reward_threshold)

            if self.config.option == 'hard':
                return reward >= threshold
            elif self.config.option == 'soft':
                return random.uniform(0, 1) < min(1., math.exp((reward - threshold) / self.config.beta))
            else:
                assert False, 'Invalid reward sampling option!'

        while (input_ids.shape[1] - prompt_len) < self.config.max_new_tokens and ~stopping_criteria(input_ids, None):

            # sample a new candidate
            candidate = input_ids.clone()
            model_kwargs = self._get_initial_cache_position(candidate, model_kwargs)
            while candidate.shape[1] - input_ids.shape[1] < 64 and ~stopping_criteria(candidate, None):
                model_inputs = self.llm.prepare_inputs_for_generation(candidate, **model_kwargs)
                outputs = self.llm(
                    **model_inputs,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.llm.config.is_encoder_decoder
                )

                if candidate.shape[1] - input_ids.shape[1] >= 4 and entropy(
                        next_token_logits).item() >= self.config.entropy_threshold:
                    del next_token_logits
                    break

                val, idx = torch.topk(next_token_logits, k=self.config.top_k, dim=-1)
                selected_idx = torch.multinomial(F.softmax(val / self.config.beta, dim=-1), num_samples=1)
                selected_token = torch.gather(idx, -1, selected_idx)
                candidate = torch.cat([candidate, selected_token], dim=-1)

            # evaluate the candidate
            responses = self.llm_tokenizer.batch_decode(candidate[:, prompt_len:], skip_special_tokens=True)
            responses = self.wrap_responses(responses)
            reward = self.scorer.get_batched_reward(
                prompt=prompt,
                responses=responses
            )

            if reward.item() > best_reward:
                best_candidate, best_reward = candidate.clone(), reward.item()

            # accept/reject the candidate
            if num_regeneration >= 10:
                best_reward = -1e34
                input_ids = best_candidate
                num_regeneration = 0
            elif accept_check(reward, candidate):
                # print("######################################")
                best_candidate, best_reward = None, -1e34
                input_ids = candidate
                num_regeneration = 0
            else:
                num_regeneration += 1
            model_kwargs = {}

        return self.llm_tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/cache"
    rm_name = "weepcat/hh_sft_RM-Gemma-2B"
    llm_name = "alignment-handbook/zephyr-7b-sft-full"
    rm = AutoModelForSequenceClassification.from_pretrained(rm_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, cache_dir=cache_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir)
    llm.cuda()
    rm.cuda()

    score_config = RMScorerConfig(
        max_length=4096,
        batch_size=32,
    )

    rm_scorer = RewardModelScorer(
        rm=rm,
        rm_tokenizer=rm_tokenizer,
        config=score_config
    )

    cards_config = CARDSGeneratorConfig(
        alpha=0.5,
        beta=0.7,
        entropy_threshold=3.0,
        reward_threshold=4.,
        top_k=10,
        option="soft",
        max_new_tokens=256
    )

    cards_generator = CARDSGenerator(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        scorer=rm_scorer,
        config=cards_config
    )

    prompt = [{"content": "Human: How do I sterilize water?", "role": "user"}]
    print(cards_generator.generate(prompt))

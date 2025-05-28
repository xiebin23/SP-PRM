from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from baselines.scorer import RewardModelScorer
from baselines.config.generation_config import ARGSGeneratorConfig
from baselines.config import RMScorerConfig
from baselines.generator.base_generator import BaseGenerator
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    EosTokenCriteria,
    StoppingCriteriaList
)

import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ARGSGenerator(BaseGenerator):
    """
    A class for generating text using language models with reward model guidance.
    """
    config: ARGSGeneratorConfig
    scorer: RewardModelScorer

    def _generate(self, prompt: List[Dict], **model_kwargs) -> str:
        """
        Generate text using the language model with reward guidance.

        Args:
            prompt: Input prompt for generation

        Returns:
            dict: Contains the generated sequence
        """
        logits_wraper = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()
        if self.config.temperature:
            logits_wraper.append(TemperatureLogitsWarper(self.config.temperature))

        if self.llm_tokenizer.eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=self.llm_tokenizer.eos_token_id))

        # Process prompt
        llm_prompt = self.process_prompt(prompt)
        inputs = self.llm_tokenizer(
            llm_prompt,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.llm.device)
        attention_mask = inputs['attention_mask'].to(self.llm.device)

        k = self.config.top_k
        w = self.config.w
        mode = self.config.mode

        if self.config.max_new_tokens:
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1) + self.config.max_new_tokens))

        model_kwargs["attention_mask"] = attention_mask
        this_peer_finished = False
        is_first = True
        prompt_len = input_ids.size(1)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while not this_peer_finished:
            model_inputs = self.llm.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.llm(
                **model_inputs,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = logits_wraper(input_ids, next_token_logits).T
            top_k_tokens_id = torch.topk(next_token_logits, k, dim=0).indices[:, 0]
            next_token_logits = next_token_logits[top_k_tokens_id]

            if is_first:
                responses = self.llm_tokenizer.batch_decode(top_k_tokens_id, skip_special_tokens=True)
            else:
                responses = self.llm_tokenizer.batch_decode(
                    torch.cat([input_ids[:, prompt_len:].expand(k, -1), top_k_tokens_id.reshape(-1, 1)], dim=-1)
                )
            responses = self.wrap_responses(responses)
            scores = self.scorer(
                prompt=prompt,
                responses=responses
            )
            is_first = False
            next_token_logits = next_token_logits + w * scores
            if mode == 1:
                next_token_id = torch.argmax(next_token_logits, dim=0)
                next_token = top_k_tokens_id[next_token_id, None]
            else:
                probs = nn.functional.softmax(next_token_logits, dim=0).squeeze()
                next_token_id = torch.multinomial(probs, num_samples=1)
                next_token = top_k_tokens_id[next_token_id, None]
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.llm.config.is_encoder_decoder
            )
            this_peer_finished = stopping_criteria(input_ids, None)
        return self.llm_tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/cache"
    rm_name = "weqweasdas/RM-Gemma-2B"
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
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
    args_config = ARGSGeneratorConfig(
        max_new_tokens=128,
        w=1.,
        top_k=10,
        mode=1
    )
    args_generator = ARGSGenerator(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        scorer=rm_scorer,
        config=args_config
    )
    prompt = [{"content": "Write a story about", "role": "user"}]
    print(args_generator.generate(prompt)[0])

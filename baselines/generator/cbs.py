from typing import List, Dict, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from baselines.scorer import RewardModelScorer
from baselines.config.generation_config import CBSGeneratorConfig
from baselines.config import RMScorerConfig
from baselines.generator.base_generator import BaseGenerator
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
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
class CBSGenerator(BaseGenerator):
    scorer: RewardModelScorer
    config: CBSGeneratorConfig
    """
    A class for generating text using language models with reward model guidance.
    """
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx)
                      for past_state in layer_past),
            )
        return reordered_past

    def _generate(self, prompt: Union[List[Dict], str], **model_kwargs):
        logits_wrapper = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()
        if self.config.temperature:
            logits_wrapper.append(TemperatureLogitsWarper(self.config.temperature))
        if self.config.top_k:
            logits_wrapper.append(TopKLogitsWarper(self.config.top_k))
        if self.config.top_p:
            logits_wrapper.append(TopPLogitsWarper(self.config.top_p))

        if self.config.max_new_tokens is None:
            stopping_criteria.append(MaxLengthCriteria(max_length=self.config.max_length))
        if self.llm_tokenizer.eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=self.llm_tokenizer.eos_token_id))

        llm_prompt = self.process_prompt(prompt)
        if self.llm_tokenizer.pad_token_id is not None:
            pad_token_id = self.llm_tokenizer.pad_token_id
        else:
            pad_token_id = self.llm_tokenizer.eos_token_id

        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        inputs = self.llm_tokenizer(
            llm_prompt,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.llm.device)
        attention_mask = inputs['attention_mask'].to(self.llm.device)

        w = self.config.beam_width
        k = self.config.successors_per_state
        l = self.config.chunk_length
        tokens_remain_per_chunk = l

        input_ids = input_ids.expand(w * k, -1)
        model_kwargs["attention_mask"] = attention_mask.expand(w * k, -1)

        if self.config.max_new_tokens:
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1) + self.config.max_new_tokens))

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        prompt_len = input_ids.size(1)
        model_kwargs = self.llm._get_initial_cache_position(input_ids, model_kwargs)
        while not this_peer_finished:
            model_inputs = self.llm.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.llm(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = logits_wrapper(input_ids, next_token_logits)

            # 采样下一个token
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # 处理已完成的序列
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            tokens_remain_per_chunk -= 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.llm.config.is_encoder_decoder
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            # torch.cuda.empty_cache()

            if tokens_remain_per_chunk <= 0 or this_peer_finished:
                tokens_remain_per_chunk = l

                responses = self.llm_tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
                responses = self.wrap_responses(responses)
                beam_scores = self.scorer(prompt, responses)
                _, beam_idx = torch.topk(beam_scores, w, dim=0, largest=True, sorted=True)
                beam_idx = beam_idx.squeeze().repeat(k)
                input_ids = input_ids[beam_idx]
                unfinished_sequences = unfinished_sequences[beam_idx]
                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = self._reorder_cache(
                        model_kwargs["past_key_values"],
                        beam_idx
                    )
                # torch.cuda.empty_cache()
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

    cbs_config = CBSGeneratorConfig(
        beam_width=2,
        successors_per_state=2,
        chunk_length=30,
        max_new_tokens=128
    )

    cbs_generator = CBSGenerator(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        scorer=rm_scorer,
        config=cbs_config
    )

    prompt = [{"content": "Write a story about", "role": "user"}]
    print(cbs_generator.generate(prompt)[0])

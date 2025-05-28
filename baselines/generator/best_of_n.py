from typing import List, Dict, Union
from baselines.scorer.reward_scorer import RewardModelScorer
from baselines.scorer.base_scorer import BaseScorer
from baselines.config.generation_config import BestOfNGeneratorConfig
from baselines.config.scorer_config import RMScorerConfig
from baselines.generator.base_generator import BaseGenerator
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch


@dataclass
class BestOfNGenerator(BaseGenerator):
    config: BestOfNGeneratorConfig
    scorer: BaseScorer

    def _generate(self, prompt: Union[List[Dict], str], **model_args) -> str:
        """
        Generate text using the language model with reward guidance.

        Args:
            prompt: Input prompt for generation

        Returns:
            dict: Contains the generated sequence
        """
        llm_prompt = self.process_prompt(prompt)
        inputs = self.llm_tokenizer(
            llm_prompt,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.llm.device)
        attention_mask = inputs['attention_mask'].to(self.llm.device)
        prompt_len = input_ids.size(1)

        output = self.llm.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            num_return_sequences=self.config.num_return_sequences
        )

        responses = self.llm_tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        responses = self.wrap_responses(responses)
        scores = self.scorer.get_batched_reward(prompt, responses)
        idx = scores.topk(1, dim=0).indices[0].squeeze()
        return self.llm_tokenizer.decode(output[idx, prompt_len:], skip_special_tokens=True)


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
    bon_config = BestOfNGeneratorConfig(
        max_new_tokens=128,
        top_k=10,
        num_return_sequences=16,
        do_sample=True
    )
    bon_generator = BestOfNGenerator(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        scorer=rm_scorer,
        config=bon_config
    )
    prompt = [{"content": "Write a story about", "role": "user"}]
    print(bon_generator.generate(prompt)[0])

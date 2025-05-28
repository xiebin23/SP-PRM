from typing import List, Dict, Union
from baselines.generator.base_generator import BaseGenerator
from baselines.config.generation_config import BaseGeneratorConfig
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class VanillaGenerator(BaseGenerator):

    def _generate(self, prompt: Union[List[Dict], str, List[List[Dict]]], **model_args) -> str:
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
        )

        return self.llm_tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    cache_dir = "/root/autodl-tmp/cache"
    rm_name = "weqweasdas/RM-Gemma-2B"
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    llm = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir)
    llm.cuda()

    base_config = BaseGeneratorConfig(
        max_new_tokens=128,
    )
    vanilla_generator = VanillaGenerator(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        config=base_config
    )
    prompt = [{"content": "Write a story about", "role": "user"}]
    print(vanilla_generator.generate(prompt)[0])

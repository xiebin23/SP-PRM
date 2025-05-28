from typing import List, Dict, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationMixin
from baselines.config.generation_config import BaseGeneratorConfig
from dataclasses import dataclass
from utils.util import apply_chat_template

@dataclass
class BaseGenerator(GenerationMixin):
    llm: PreTrainedModel
    llm_tokenizer: PreTrainedTokenizer
    config: BaseGeneratorConfig

    def process_prompt(self, prompt: Union[List[Dict], str]):
        if isinstance(prompt, list):
            if self.llm_tokenizer.chat_template is not None:
                if self.llm_tokenizer.bos_token:
                    prompt = self.llm_tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    ).replace(self.llm_tokenizer.bos_token, "")
                else:
                    prompt = self.llm_tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            else:
                prompt = apply_chat_template(prompt)

        return prompt

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.llm.prepare_inputs_for_generation(input_ids, **model_kwargs)

    @staticmethod
    def wrap_responses(responses: List[str]) -> List[List[Dict]]:
        responses = [[{"content": response, "role": "assistant"}] for response in responses]
        return responses

    @torch.no_grad()
    def generate(self, prompt: Union[List[Dict], str], return_dict_in_generate: bool = True) -> str:
        output = self._generate(prompt)
        return output

    def _generate(self, prompt: Union[List[Dict], str], **model_args) -> str:
        raise NotImplementedError

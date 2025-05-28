import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from baselines.scorer import RewardModelScorer
from baselines.config.scorer_config import RMScorerConfig
from baselines.config.generation_config import BaseGeneratorConfig, ARGSGeneratorConfig, BestOfNGeneratorConfig, CBSGeneratorConfig
from baselines.generator import ARGSGenerator, CBSGenerator, BestOfNGenerator, VanillaGenerator

# Setting
rm_name = "weqweasdas/RM-Gemma-2B"
llm_name = "alignment-handbook/zephyr-7b-sft-full"
dataset_name = "weepcat/hh-rlhf-eval"
top_p = 1.
top_k = 10
max_new_tokens = 128
w_of_args = 1.0
k_of_cbs = 2
w_of_cbs = 2
l_of_cbs = 30
n_of_bon = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

# load dataset
eval_ds = load_dataset(dataset_name, split="eval")

# load model and tokenizer
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_name,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
).cuda()

rm_tokenizer = AutoTokenizer.from_pretrained(
    rm_name
)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
).cuda()

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name
)

score_config = RMScorerConfig()
rm_scorer = RewardModelScorer(
    rm=rm,
    rm_tokenizer=rm_tokenizer,
    config=score_config
)

# Generator Config
vanilla_greedy_config = BaseGeneratorConfig(
    max_new_tokens=max_new_tokens,
)

vanilla_top_k_config = BaseGeneratorConfig(
    max_new_tokens=max_new_tokens,
    do_sample=True,
    top_k=top_k
)

vanilla_top_p_config = BaseGeneratorConfig(
    max_new_tokens=max_new_tokens,
    do_sample=True,
    top_p=top_p
)

bon_config = BestOfNGeneratorConfig(
    num_return_sequences=n_of_bon,
    max_new_tokens=max_new_tokens,
    do_sample=True
)

args_greedy_config = ARGSGeneratorConfig(
    top_k=top_k,
    max_new_tokens=max_new_tokens,
    mode=1,
    w=w_of_args,
)

args_stochastic_config = ARGSGeneratorConfig(
    top_k=top_k,
    max_new_tokens=max_new_tokens,
    mode=2,
    w=w_of_args,
)

cbs_config = CBSGeneratorConfig(
    max_new_tokens=max_new_tokens,
    beam_width=2,
    successors_per_state=2,
    chunk_length=30
)
generators = {}

generators['vanilla_greedy'] = VanillaGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    config=vanilla_greedy_config
)

generators['vanilla_top_k'] = VanillaGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    config=vanilla_top_k_config
)

generators['vanilla_top_p'] = VanillaGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    config=vanilla_top_p_config
)

generators['best_of_n'] = BestOfNGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    scorer=rm_scorer,
    config=bon_config
)

generators['args_greedy'] = ARGSGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    scorer=rm_scorer,
    config=args_greedy_config
)

generators['args_stochastic'] = ARGSGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    scorer=rm_scorer,
    config=args_stochastic_config
)

generators['cbs'] = CBSGenerator(
    llm=llm,
    llm_tokenizer=llm_tokenizer,
    scorer=rm_scorer,
    config=cbs_config
)


if __name__ == "__main__":
    prompt = eval_ds[0]['prompt']
    vanilla_result = generators['vanilla_greedy'].generate(prompt)
    print("Vanilla result: ", vanilla_result)
    bon_result = generators['best_of_n'].generate(prompt)
    print("Best of N result: ", bon_result)
    args_greedy_result = generators['args_greedy'].generate(prompt)
    print("ARGS Greedy result: ", args_greedy_result)


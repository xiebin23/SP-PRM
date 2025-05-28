from utils.model_factory import ModelFactory
from utils.tokenizer_factory import TokenizerFactory
from utils.util import wrap_model_name
import json
import torch

llm = "meta-llama/Llama-3.2-3B-Instruct"
# "meta-llama/Llama-3.2-3B-Instruct"
# 'alignment-handbook/zephyr-7b-sft-full'
rm = "weepcat/summarization_sft_reward-model-deberta-v3-large-v2"
partial_process_method = "random_length"
method = "best_of_n_partial_16"
emb_model = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
model = ModelFactory.load_embedding(
    emb_model,
    cache_dir='/root/autodl-tmp/cache'
)
tokenizer = TokenizerFactory.load_tokenizer(
    emb_model,
    cache_dir='/root/autodl-tmp/cache'
)
# token
# "args_full_greedy_w_1.5_top_k_40"
# "args_partial_greedy_w_2.5_top_k_10"
# "cbs_full_w_6_k_2_l_1"
# "cbs_partial_w_6_k_2_l_1"
# chunk
# "cbs_full_w_8_k_6_l_10"
# "cbs_full_w_6_k_8_l_30"
# "cbs_full_w_8_k_8_l_50"
# "cbs_partial_w_8_k_6_l_10"
# "cbs_partial_w_8_k_6_l_30"
# "cbs_partial_w_8_k_8_l_50"
# sentence
# cards_full_rt_6.0_top_k_30_et_3.0
# cards_full_rt_6.0_top_k_30_et_5.0
# cards_full_rt_6.0_top_k_30_et_7.0
# cards_partial_rt_6.0_top_k_30_et_3.0
# cards_partial_rt_5.0_top_k_20_et_5.0
# cards_partial_rt_5.0_top_k_40_et_7.0
# response
# "best_of_n_full_16"
# "best_of_n_full_64"
# "best_of_n_partial_16"
# "best_of_n_partial_64"


@torch.no_grad()
def compute_coherence(model, tokenizer, prompt, method_response):
    prompt_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    for key in prompt_inputs.keys():
        prompt_inputs[key] = prompt_inputs[key].cuda()
    prompt_embedding = model(**prompt_inputs, output_hidden_states=True, return_dict=True).pooler_output
    response_inputs = tokenizer(method_response, padding=True, truncation=True, return_tensors="pt")
    for key in response_inputs.keys():
        response_inputs[key] = response_inputs[key].cuda()
    response_embedding = model(**response_inputs, output_hidden_states=True, return_dict=True).pooler_output
    coherence_score = torch.cosine_similarity(prompt_embedding, response_embedding)
    return coherence_score.item()


# 1. 读取 json 文件
with open(
        f"./results/evaluate/{wrap_model_name(llm)}/{wrap_model_name(rm)}/{partial_process_method}/baseline_results_sample_100_seed_42.json") as f:
    data = json.load(f)

# 2. 提取 prompt, vanilla_greedy 和 method 的 response
prompts = list(data.keys())
scores = []
for i, prompt in enumerate(prompts):
    method_response = data[prompt][method]
    # 计算coherence
    coherence_score = compute_coherence(model, tokenizer, prompt, method_response)
    scores.append(coherence_score)

print(f"Average diversity score: {round(sum(scores) / len(scores), 4)}")

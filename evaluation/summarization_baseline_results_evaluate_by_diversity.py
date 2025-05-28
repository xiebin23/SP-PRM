from utils.tokenizer_factory import TokenizerFactory
from nltk.util import ngrams
from utils.util import wrap_model_name
import json


# Function to compute diversity score using transformer tokenizer
def compute_diversity(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    diversity_score = 1.0

    for n in range(2, 5):
        n_grams = list(ngrams(tokens, n))
        unique_n_grams = len(set(n_grams))
        total_n_grams = len(n_grams)
        try:
            diversity_score *= unique_n_grams / total_n_grams
        except Exception as e:
            print(text)

    return diversity_score


llm = "meta-llama/Llama-3.2-3B-Instruct"
# "meta-llama/Llama-3.2-3B-Instruct"
# 'alignment-handbook/zephyr-7b-sft-full'
rm = "weepcat/summarization_sft_reward-model-deberta-v3-large-v2"
partial_process_method = "random_length"
method = "best_of_n_full_64"
tokenizer = TokenizerFactory.load_tokenizer(
    llm,
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

# 1. 读取 json 文件
with open(
        f"./results/evaluate/{wrap_model_name(llm)}/{wrap_model_name(rm)}/{partial_process_method}/baseline_results_sample_100_seed_42.json") as f:
    data = json.load(f)

# 2. 提取 prompt, vanilla_greedy 和 method 的 response
prompts = list(data.keys())
scores = []
for i, prompt in enumerate(prompts):
    method_response = data[prompt][method]
    # 计算diversity
    diversity_score = compute_diversity(tokenizer, method_response)
    scores.append(diversity_score)

print(f"Average diversity score: {round(sum(scores) / len(scores), 4)}")

# SPRM

## Introduction

Welcome to the SPRM project repository. In this repository, you will find the code and supplementary materials related to our research.

## Installation

To set up the project locally, please follow the instructions below:

1. Clone the repository:

   ```sh
   git clone https://github.com/WeepCat/SPRM.git
   cd SPRM
   ```

2. Create and activate a virtual environment:

   ```sh
   conda create -n sprm python=3.10
   conda activate sprm
   ```

3. Install the required dependencies:
   ```sh
   pip install transformers trl datasets
   ```

## Example

After setting up the environment, you can run the experiments and analysis scripts as follows:

1. **Data Construction:**

   ```sh
   python partial_data_construct/hh_data_process_partial.py
   ```

2. **Reward Modeling:**

   ```sh
   python reward_modeling/hh_base_model_sft.py
   python reward_modeling/hh_partial_reward_model_with_weights.py
   ```

3. **Baselines:**

   ```sh
   python generation/hh_collect_baseline_results.py
   ```

4. **Evaluation:**
   ```sh
   python evaluation/hh_baseline_results_evaluate_by_rm.py
   python evaluation/hh_baseline_results_evaluate_by_gpt.py
   python evaluation/hh_baseline_results_evaluate_by_diversity.py
   python evaluation/hh_baseline_results_evaluate_by_coherence.py
   ```

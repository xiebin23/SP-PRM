from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizer,
    TrainerCallback,
)
import torch
import torch.nn as nn
import evaluate
from utils.util import apply_chat_template, wrap_model_name
acc = evaluate.load("accuracy", cache_dir="/root/autodl-tmp/cache/")


def compute_metrics(eval_pred, compute_result=False):
    # result = {}
    # pos_predictions_scores = eval_pred.predictions[0]
    # neg_predictions_scores = eval_pred.predictions[1]
    # # We assume that the first sample is preferred by default in groundtruth
    # result['accuracy'] = np.sum(
    #     pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    # return result

    if not compute_result:
        # 每个批次的预测结果添加到 metric 中
        pos_scores, neg_scores = eval_pred.predictions
        labels = (pos_scores > neg_scores)
        acc.add_batch(predictions=labels, references=[1] * len(labels))
        return {}

    else:
        # 汇总并返回最终结果
        result = acc.compute()
        return {"accuracy": result["accuracy"]}


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizer
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            if self.tokenizer.chat_template is not None:
                positive = self.tokenizer.apply_chat_template(
                    feature['chosen'], tokenize=False,
                    add_generation_prompt=False
                ).replace(self.tokenizer.bos_token, "")

                negative = self.tokenizer.apply_chat_template(
                    feature['rejected'], tokenize=False,
                    add_generation_prompt=False
                ).replace(self.tokenizer.bos_token, "")
            else:
                positive = apply_chat_template(feature['chosen'])
                negative = apply_chat_template(feature['rejected'])

            positive = self.tokenizer(positive, truncation=True, padding=True)
            negative = self.tokenizer(negative, truncation=True, padding=True)
            merged_features.append(
                {
                    "input_ids": positive["input_ids"],
                    "attention_mask": positive["attention_mask"],
                }
            )

            merged_features.append(
                {
                    "input_ids": negative["input_ids"],
                    "attention_mask": negative["attention_mask"],
                }
            )

        batch = self.tokenizer.pad(
            merged_features,
            return_tensors="pt",
        )

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


@dataclass
class MaskWeightedRewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizer
    ref_model: str
    mask: str
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        weights = []
        column_name = wrap_model_name(self.ref_model) + ("_masked_weights" if self.mask == "mask" else "_weights")
        is_eval = True
        for feature in features:
            weight = getattr(feature, column_name, None)
            if weight is not None:
                weights.append([feature[column_name]])
                is_eval = False
            if self.tokenizer.chat_template is not None:
                positive = self.tokenizer.apply_chat_template(
                    feature['chosen'], tokenize=False,
                    add_generation_prompt=False
                ).replace(self.tokenizer.bos_token, "")

                negative = self.tokenizer.apply_chat_template(
                    feature['rejected'], tokenize=False,
                    add_generation_prompt=False
                ).replace(self.tokenizer.bos_token, "")
            else:
                positive = apply_chat_template(feature['chosen'])
                negative = apply_chat_template(feature['rejected'])
            positive = self.tokenizer(positive, truncation=True, padding=True)
            negative = self.tokenizer(negative, truncation=True, padding=True)
            merged_features.append(
                {
                    "input_ids": positive["input_ids"],
                    "attention_mask": positive["attention_mask"],
                }
            )
            merged_features.append(
                {
                    "input_ids": negative["input_ids"],
                    "attention_mask": negative["attention_mask"],
                }
            )

        batch = self.tokenizer.pad(
            merged_features,
            return_tensors="pt",
        )

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }

        if not is_eval:
            batch["weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardTrainerWithWeight(Trainer):
    tokenizer: PreTrainedTokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        if inputs.get('weights') is not None:
            loss = (-nn.functional.logsigmoid(rewards_j - rewards_k) * inputs['weights']).mean()
        else:
            loss = (-nn.functional.logsigmoid(rewards_j - rewards_k)).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

    # def save_model(self, output_dir=None, _internal_call: bool = False):
    #     super().save_model(output_dir)
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)

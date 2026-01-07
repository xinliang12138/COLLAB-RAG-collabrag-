import re, json, string
from tqdm import tqdm
import numpy as np
import os
import json
import sys

# 添加项目根目录到路径，以便导入common_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common_utils


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['answers'], item["rationale"]))

        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit)


# 新增：计算辩论影响力指标
def compute_debate_metrics(data):
    """计算辩论相关的评估指标
    Args:
        data: 包含辩论相关字段的数据结果列表
    Returns:
        包含辩论指标的字典
    """
    if 'debate_summary' not in data[0]:
        return {}

    # 计算有多少样本达成共识
    consensus_count = 0
    # 统计辩论影响了最终答案的数量
    opinion_changed = 0
    # 不同轮次辩论的数量统计
    round_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for item in data:
        if 'debate_summary' not in item:
            continue

        # 使用正则表达式提取辩论轮数
        debate_summary = item.get('debate_summary', '')
        round_match = re.findall(r'Round (\d+) Debate', debate_summary)

        if round_match:
            max_round = max([int(r) for r in round_match])
            if max_round in round_counts:
                round_counts[max_round] += 1
        else:
            # 如果没有找到轮数标记，默认为第1轮
            round_counts[1] += 1

        # 检查是否达成共识
        if 'reached consensus' in debate_summary.lower() or 'agreeing on the answer' in debate_summary.lower():
            consensus_count += 1

        # 检查辩论是否改变了原始意见
        if 'changed their initial view' in debate_summary.lower() or 'changed their mind' in debate_summary.lower():
            opinion_changed += 1

    total_samples = len([item for item in data if 'debate_summary' in item])
    if total_samples == 0:
        return {}

    debate_metrics = {
        "consensus_rate": consensus_count / total_samples * 100 if total_samples > 0 else 0,
        "opinion_change_rate": opinion_changed / total_samples * 100 if total_samples > 0 else 0,
        "round_distribution": {k: v / total_samples * 100 for k, v in round_counts.items() if v > 0},
        "avg_rounds": sum(k * v for k, v in round_counts.items()) / total_samples if total_samples > 0 else 0,
        "samples_with_debate": total_samples
    }

    return debate_metrics


def get_metrics(data, save_dir=None, is_asqa=False, generation_params=None):
    idx = 0
    num_accurate = 0
    print('Evaluating results...')

    # 无论是什么数据集，都计算样本数
    idx = len(data)  # 添加这一行来获取正确的样本数

    if is_asqa:
        rationale_str_em, str_em_hit = compute_str_em(data)
    else:
        for d in tqdm(data):
            # idx += 1  # 删除或注释这一行，因为我们已经设置了 idx = len(data)
            is_accurate = exact_presence(d['answers'], d['rationale'])
            num_accurate += 1 if is_accurate else 0

    # 准备评估结果字典
    if is_asqa:
        print(f"Rationale EM: {rationale_str_em:.1f}%")
        eval_result = {
            "EM": rationale_str_em,
            "EM_HIT": str_em_hit,
            "num_examples": idx
        }
    else:
        accuracy = num_accurate / idx * 100 if idx > 0 else 0
        print(f"Accuracy: {accuracy:.1f}%")
        eval_result = {
            "accuracy": accuracy,
            "num_examples": idx
        }

    # 添加生成参数到评估结果
    if generation_params:
        eval_result["generation_params"] = {
            "seed": generation_params.get("seed", 42),
            "temperature": generation_params.get("temperature", 0.0),
            "top_k": generation_params.get("top_k", 50),
            "top_p": generation_params.get("top_p", 0.95),
            "batch_size": generation_params.get("batch_size", 16),
            "use_multiagent": generation_params.get("use_multiagent", False)
        }

        # 如果使用了多智能体，添加多智能体参数
        if generation_params.get("use_multiagent") and generation_params.get("multiagent_params"):
            eval_result["generation_params"]["multiagent_params"] = generation_params["multiagent_params"]

        # 打印生成参数
        print(f"Generation parameters:")
        for k, v in eval_result["generation_params"].items():
            if k != "multiagent_params":
                print(f"  {k}: {v}")
            elif v:
                print(f"  {k}:")
                for mk, mv in v.items():
                    print(f"    {mk}: {mv}")

    # 如果启用了多智能体，计算辩论相关指标
    if generation_params and generation_params.get("use_multiagent", False):
        debate_metrics = compute_debate_metrics(data)
        if debate_metrics:
            eval_result["debate_metrics"] = debate_metrics
            print("\nDebate Metrics:")
            for k, v in debate_metrics.items():
                if k != "round_distribution":
                    print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
                else:
                    print(f"  {k}:")
                    for rk, rv in v.items():
                        print(f"    {rk} rounds: {rv:.1f}%")

    # 保存到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(eval_result, indent=2, fp=f)
        print(f"Metrics saved to {save_dir}/metrics.json")

    return eval_result
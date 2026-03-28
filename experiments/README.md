# Experiments

评估实验脚本。

## 快速开始

```bash
# 实验1: 学术问答质量评估
python evaluate_experiment_01.py \
    --queries data/rag_queries.jsonl \
    --out-dir results/exp01_academic_agent

# 实验3: 任务尝试次数与成功率分析
python evaluate_experiment_03.py \
    --scores output/training_free_grpo/trajectory_scores.jsonl \
    --out-dir results/exp03
```

## 配置文件 (config.yaml)

```yaml
judge:
  api_base: "https://api.deepseek.com"
  api_key: "sk-xxx"
  model: "deepseek-chat"
  timeout: 120
```

## 实验说明

| 实验 | 目标 | 数据来源 |
|------|------|---------|
| exp1 | 学术问答质量（Relevance/Accuracy/Completeness/Citation） | `data/rag_queries.jsonl` |
| exp3 | 尝试次数与成功率关系，边际收益分析 | `trajectory_scores.jsonl` |

## 输出

每个实验结果包含：
- `details.jsonl` - 详细评估结果
- `summary.json` - 汇总统计
- `figures/` - 图表目录

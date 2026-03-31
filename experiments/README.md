# Experiments

评估实验脚本。

## 快速开始

```bash
# 实验一: 学术搜索 RAG 质量评估
python academic_rag_eval.py \
    --queries data/rag_queries.jsonl \
    --out-dir results/academic_rag_agent

# 实验一 Baseline（无搜索）
python academic_rag_eval.py \
    --queries data/rag_queries.jsonl \
    --out-dir results/academic_baseline \
    --baseline

# 实验一: 对比结果（自动生成）
python academic_rag_compare.py

# 实验三: 任务尝试次数与成功率分析
python task_attempt_analysis_eval.py \
    --scores output/training_free_grpo/trajectory_scores.jsonl \
    --out-dir results/task_attempt_analysis
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

| 实验 | 目标 | 数据来源 | 结果 |
|------|------|---------|------|
| academic_rag | 学术问答质量（Relevance/Accuracy/Completeness/Citation） | `data/rag_queries.jsonl` | 综合 75.1%，引用提升最显著 +34.9pp |
| academic_baseline | 无搜索 Baseline 对照 | `data/rag_queries.jsonl` | 综合 66.7% |
| task_attempt_analysis | 尝试次数与成功率关系，边际收益分析 | `trajectory_scores.jsonl` | 1次 56.3%，4次可达 87.2% |

## 输出

每个实验结果包含：
- `details.jsonl` - 详细评估结果
- `summary.json` - 汇总统计
- `figures/` - 图表目录
- `academic_rag_comparison/` - 额外生成对比图表

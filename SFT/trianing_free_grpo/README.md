# trianing_free_grpo

Training-Free Group Relative Policy Optimization 的最小实现，包含 4 个阶段：

1. 采集轨迹（同一个 prompt 多次采样）
2. 用原始模型对轨迹打分
3. 用原始模型总结高低分轨迹，提炼经验
4. 将经验写入外部存储文件

## 运行

```bash
python SFT/trianing_free_grpo/pipeline.py \
  --base_url http://localhost:1234/v1 \
  --model qwen-qwen3-14b-mlx@4bit \
  --api_key no_need \
  --prompts SFT/data.txt \
  --output_dir output/trianing_free_grpo \
  --samples_per_prompt 3 \
  --steps collect,score,summarize
```

## 输出

- `output/trianing_free_grpo/trajectories.jsonl`: 采集轨迹
- `output/trianing_free_grpo/trajectory_scores.jsonl`: 轨迹评分
- `output/trianing_free_grpo/external_memory.json`: 外部经验库（结构化）
- `output/trianing_free_grpo/external_memory.md`: 外部经验库（可读版）

## 分步执行

```bash
# 仅采样
python SFT/trianing_free_grpo/pipeline.py --steps collect

# 在已有采样上打分
python SFT/trianing_free_grpo/pipeline.py --steps score

# 在已有评分上总结经验
python SFT/trianing_free_grpo/pipeline.py --steps summarize
```

脚本默认支持断点续跑：已完成的 `prompt_id + attempt_id` 不会重复处理。

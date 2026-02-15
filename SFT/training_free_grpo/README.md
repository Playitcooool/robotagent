# training_free_grpo

现在只保留 3 个可直接运行的单文件脚本，彼此不依赖本地模块：

- `SFT/training_free_grpo/collect.py`
- `SFT/training_free_grpo/score.py`
- `SFT/training_free_grpo/summarize.py`

## 1) 采集轨迹（Deep Agent）

```bash
python SFT/training_free_grpo/collect.py \
  --base_url http://localhost:1234/v1 \
  --model lmstudio-community-qwen3-4b-instruct-2507-mlx \
  --api_key no_need \
  --prompts SFT/data.txt \
  --output_path output/training_free_grpo/trajectories.jsonl \
  --samples_per_prompt 3
```

## 2) 轨迹打分

```bash
python SFT/training_free_grpo/score.py \
  --base_url http://localhost:1234/v1 \
  --model lmstudio-community-qwen3-4b-instruct-2507-mlx \
  --api_key no_need \
  --trajectory_path output/training_free_grpo/trajectories.jsonl \
  --score_path output/training_free_grpo/trajectory_scores.jsonl
```

## 3) 经验总结并外部存储

```bash
python SFT/training_free_grpo/summarize.py \
  --base_url http://localhost:1234/v1 \
  --model lmstudio-community-qwen3-4b-instruct-2507-mlx \
  --api_key no_need \
  --score_path output/training_free_grpo/trajectory_scores.jsonl \
  --memory_json_path output/training_free_grpo/external_memory.json \
  --memory_md_path output/training_free_grpo/external_memory.md
```

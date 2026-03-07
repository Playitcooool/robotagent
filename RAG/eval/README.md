# Robotics QA Eval Queries (5Y)

文件：`robotics_qa_queries_5y.jsonl`

## 字段

- `id`: 查询编号
- `query`: 用户问题
- `category`: 主题类别（manipulation/navigation/safety 等）
- `type`: 问题类型（fact/compare/reasoning/survey/synthesis）
- `year_scope`: 建议检索年份范围（当前为 `2021-2026`）
- `need_citation`: 是否要求回答附出处（建议全部为 true）

## 使用建议

1. 让系统逐条回答这些 query。
2. 记录回答与引用（最好提取 `[title](url)`）。
3. 用外部 LLM judge 评估：
   - factuality
   - grounding
   - completeness
   - citation quality
4. 按 `category` 与 `type` 分组统计均分与方差。

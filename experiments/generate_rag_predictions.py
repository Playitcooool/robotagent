#!/usr/bin/env python3
"""
生成RAG预测数据

用于实验2：使用RAG系统回答问题，然后评估回答质量。

使用方式:
    python generate_rag_predictions.py \
        --queries data/rag_queries.jsonl \
        --out data/predictions.jsonl

注意：需要确保后端服务正在运行。
"""

import argparse
import json
import yaml
from pathlib import Path
import requests


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_queries(path: Path):
    """加载查询数据"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def query_chat_api(question: str, api_base: str, auth_token: str = "") -> dict:
    """调用聊天API获取RAG回答"""
    try:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = requests.post(
            f"{api_base}/api/chat/send",
            json={
                "message": question,
                "enabled_tools": []  # 根据需要配置
            },
            headers=headers,
            timeout=120,
            stream=False
        )
        if response.ok:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "answer": ""}
    except Exception as e:
        return {"error": str(e), "answer": ""}


def main():
    parser = argparse.ArgumentParser(description="生成RAG预测数据")
    parser.add_argument("--queries", required=True, help="查询数据JSONL文件")
    parser.add_argument("--out", required=True, help="输出预测结果文件")
    parser.add_argument("--api-base", default="http://localhost:8000", help="后端API地址")
    parser.add_argument("--auth-token", default="", help="认证Token")
    parser.add_argument("--limit", type=int, default=0, help="限制查询数量，0表示全部")
    args = parser.parse_args()

    # 加载查询
    queries_path = Path(args.queries)
    queries = load_queries(queries_path)
    if args.limit > 0:
        queries = queries[:args.limit]

    print(f"Loaded {len(queries)} queries")
    print(f"Using API: {args.api_base}")

    # 生成预测
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(queries):
        question = q.get("query", q.get("question", ""))
        print(f"[{i+1}/{len(queries)}] Querying: {question[:50]}...")

        result = query_chat_api(question, args.api_base, args.auth_token)

        output = {
            "id": q.get("id", f"rag_{i+1:03d}"),
            "question": question,
            "answer": result.get("answer", result.get("text", "")),
            "references": result.get("references", []),
            "error": result.get("error")
        }
        results.append(output)

        # 写入文件
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"\nDone! Results saved to {out_path}")
    print(f"Total: {len(results)}, Errors: {sum(1 for r in results if r.get('error'))}")


if __name__ == "__main__":
    main()

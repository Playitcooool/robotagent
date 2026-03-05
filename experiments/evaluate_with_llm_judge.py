import argparse
import json
import os
from pathlib import Path

import requests


SYSTEM_PROMPT = (
    "You are a strict evaluator for QA quality. "
    "Score with concise JSON only. "
    "Do not include markdown or explanation outside JSON."
)


def load_jsonl(path: Path):
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


def call_judge(base_url: str, api_key: str, model: str, user_prompt: str, timeout: float = 60.0):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def build_single_prompt(row):
    return (
        "Evaluate the answer quality.\n"
        "Return strict JSON with keys: relevance,factuality,grounding,actionability,overall,verdict.\n"
        "Each score is 1-5 integer.\n\n"
        f"Prompt:\n{row.get('prompt','')}\n\n"
        f"Answer:\n{row.get('answer','')}\n\n"
        f"References:\n{json.dumps(row.get('references', []), ensure_ascii=False)}"
    )


def build_pairwise_prompt(row_a, row_b):
    return (
        "Compare answer A and answer B for the same prompt.\n"
        "Return strict JSON with keys: winner,score_a,score_b,reason.\n"
        "winner must be one of: A,B,TIE.\n"
        "score_a/score_b are 1-5 overall quality.\n\n"
        f"Prompt:\n{row_a.get('prompt','')}\n\n"
        f"Answer A:\n{row_a.get('answer','')}\n\n"
        f"Answer B:\n{row_b.get('answer','')}\n"
    )


def summarize(rows, pairwise=False):
    if not rows:
        return {"count": 0}
    if pairwise:
        wins = {"A": 0, "B": 0, "TIE": 0}
        sa = []
        sb = []
        for r in rows:
            j = r.get("judgment", {})
            wins[str(j.get("winner", "TIE")).upper()] = wins.get(str(j.get("winner", "TIE")).upper(), 0) + 1
            sa.append(float(j.get("score_a", 0)))
            sb.append(float(j.get("score_b", 0)))
        return {
            "count": len(rows),
            "wins": wins,
            "avg_score_a": sum(sa) / max(1, len(sa)),
            "avg_score_b": sum(sb) / max(1, len(sb)),
        }
    keys = ["relevance", "factuality", "grounding", "actionability", "overall"]
    acc = {k: 0.0 for k in keys}
    for r in rows:
        j = r.get("judgment", {})
        for k in keys:
            acc[k] += float(j.get(k, 0))
    return {"count": len(rows), **{k: acc[k] / len(rows) for k in keys}}


def main():
    parser = argparse.ArgumentParser(description="Evaluate answers using external LLM judge.")
    parser.add_argument("--predictions", required=True, help="JSONL file with id/prompt/answer/references")
    parser.add_argument("--baseline", default="", help="Optional JSONL for pairwise A/B judge")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.base_url or not args.model:
        raise ValueError("base-url and model are required (or set JUDGE_BASE_URL / JUDGE_MODEL)")

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_a = load_jsonl(pred_path)
    if args.limit and args.limit > 0:
        rows_a = rows_a[: args.limit]

    pairwise = bool(args.baseline)
    rows_out = []
    if pairwise:
        rows_b = load_jsonl(Path(args.baseline))
        by_id_b = {str(r.get("id")): r for r in rows_b}
        for row_a in rows_a:
            rid = str(row_a.get("id", ""))
            row_b = by_id_b.get(rid)
            if row_b is None:
                continue
            prompt = build_pairwise_prompt(row_a, row_b)
            judgment = call_judge(args.base_url, args.api_key, args.model, prompt)
            rows_out.append({"id": rid, "judgment": judgment})
    else:
        for row in rows_a:
            rid = str(row.get("id", ""))
            prompt = build_single_prompt(row)
            judgment = call_judge(args.base_url, args.api_key, args.model, prompt)
            rows_out.append({"id": rid, "judgment": judgment})

    summary = summarize(rows_out, pairwise=pairwise)
    with (out_dir / "judgments.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

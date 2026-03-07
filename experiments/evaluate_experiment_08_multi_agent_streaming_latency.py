import argparse
from pathlib import Path

from experiment_utils import dump_json, dump_jsonl, load_jsonl, maybe_mean, ratio


VISIBLE_TEXT_TYPES = {"delta", "thinking", "status"}


def extract_events(row):
    events = row.get("events")
    return events if isinstance(events, list) else []


def get_send_ts(row, events):
    send_ts = row.get("send_ts")
    if isinstance(send_ts, (int, float)):
        return float(send_ts)
    if events:
        first_ts = events[0].get("timestamp")
        if isinstance(first_ts, (int, float)):
            return float(first_ts)
    return 0.0


def evaluate_row(row):
    events = extract_events(row)
    send_ts = get_send_ts(row, events)
    first_status = None
    first_visible_text = None
    final_done = None
    visible_sources = set()

    for event in events:
        event_type = str(event.get("type", "")).lower()
        ts = event.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        source = str(event.get("source", "")).strip().lower()
        text = str(event.get("text") or "")
        if event_type == "status" and first_status is None:
            first_status = float(ts)
        if event_type in VISIBLE_TEXT_TYPES and text.strip():
            visible_sources.add(source or "main")
            if first_visible_text is None:
                first_visible_text = float(ts)
        if event_type == "done":
            final_done = float(ts)

    empty_wait = 1 if final_done is not None and first_visible_text is None else 0
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "t_first_status": (first_status - send_ts) if first_status is not None and send_ts else None,
        "t_first_visible_text": (
            first_visible_text - send_ts if first_visible_text is not None and send_ts else None
        ),
        "t_final_answer": (final_done - send_ts) if final_done is not None and send_ts else None,
        "empty_wait": empty_wait,
        "visible_sources": sorted(source for source in visible_sources if source),
        "agent_visibility_count": len(visible_sources),
    }


def summarize(rows):
    first_status = [row["t_first_status"] for row in rows if isinstance(row["t_first_status"], (int, float))]
    first_text = [row["t_first_visible_text"] for row in rows if isinstance(row["t_first_visible_text"], (int, float))]
    final_answer = [row["t_final_answer"] for row in rows if isinstance(row["t_final_answer"], (int, float))]
    return {
        "count": len(rows),
        "avg_t_first_status": maybe_mean(first_status),
        "avg_t_first_visible_text": maybe_mean(first_text),
        "avg_t_final_answer": maybe_mean(final_answer),
        "empty_wait_ratio": ratio(sum(row["empty_wait"] for row in rows), len(rows)),
        "avg_agent_visibility_count": maybe_mean(row["agent_visibility_count"] for row in rows),
        "agent_visibility_coverage": ratio(
            sum(1 for row in rows if row["agent_visibility_count"] > 0),
            len(rows),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-agent streaming latency.")
    parser.add_argument("--input", required=True, help="JSONL with timestamped NDJSON events")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    results = [evaluate_row(row) for row in rows]
    summary = summarize(results)

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", results)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()


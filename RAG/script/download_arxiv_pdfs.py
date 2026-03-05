import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import arxiv


DEFAULT_CATEGORIES = ["cs.RO", "cs.CV", "cs.LG", "cs.AI"]
DEFAULT_INCLUDE_PHRASES = [
    "robot",
    "robotics",
    "manipulation",
    "motion planning",
    "trajectory optimization",
    "sim2real",
    "domain randomization",
    "SLAM",
    "visual odometry",
    "mobile robot",
    "locomotion",
    "grasp",
]
DEFAULT_EXCLUDE_PHRASES = [
    "medical imaging",
    "protein",
    "quantum",
    "LLM jailbreak",
    "cryptography",
]
DEFAULT_TOPIC_BUCKETS = {
    "SLAM_Perception": ["slam", "visual odometry", "localization", "mapping", "3d perception"],
    "Manipulation_Grasping": ["manipulation", "grasp", "dexterous", "pick and place"],
    "Motion_Planning_Control": ["motion planning", "trajectory optimization", "mpc", "control", "locomotion"],
    "Simulation_Sim2Real": ["simulation", "sim2real", "domain randomization", "pybullet", "gazebo"],
}


def _safe_paper_id(paper) -> str:
    """Convert arXiv id to filesystem-safe basename."""
    raw = paper.get_short_id()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def _contains_any(text: str, phrases) -> bool:
    t = (text or "").lower()
    return any((p or "").lower() in t for p in phrases)


def _paper_matches_filters(paper, include_phrases, exclude_phrases) -> bool:
    title = (paper.title or "").lower()
    abstract = (paper.summary or "").lower()
    bag = f"{title}\n{abstract}"

    if include_phrases and not _contains_any(bag, include_phrases):
        return False
    if exclude_phrases and _contains_any(bag, exclude_phrases):
        return False
    return True


def _build_query(categories, include_phrases):
    cat_clause = " OR ".join([f"cat:{c}" for c in categories])
    kw_clause = " OR ".join(
        [f'ti:"{k}" OR abs:"{k}"' for k in include_phrases]
    )
    return f"({cat_clause}) AND ({kw_clause})"


def _infer_topics(text: str, topic_buckets: dict[str, list[str]]):
    bag = (text or "").lower()
    hits = []
    for name, kws in topic_buckets.items():
        if any((kw or "").lower() in bag for kw in kws):
            hits.append(name)
    return hits


def _paper_meta(paper, pdf_path, topic_buckets):
    abstract = paper.summary or ""
    topic_hits = _infer_topics(f"{paper.title}\n{abstract}", topic_buckets)
    year = ""
    if getattr(paper, "published", None):
        try:
            year = str(paper.published.year)
        except Exception:
            year = str(paper.published)[:4]
    categories = list(paper.categories or [])
    return {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": [a.name for a in (paper.authors or [])],
        "published": str(paper.published) if getattr(paper, "published", None) else "",
        "updated": str(paper.updated) if getattr(paper, "updated", None) else "",
        "year": year,
        "categories": categories,
        "primary_category": categories[0] if categories else "",
        "topics": topic_hits,
        "pdf_url": paper.pdf_url,
        "pdf_path": pdf_path,
    }


def download_paper_pdf(paper, save_dir, topic_buckets):
    paper_id = _safe_paper_id(paper)
    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    # Skip if exists and reasonably large.
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 100 * 1024:
        return True, f"[Skip] {pdf_path}", _paper_meta(paper, pdf_path, topic_buckets)

    try:
        paper.download_pdf(filename=pdf_path)
        return True, f"[OK] {pdf_path}", _paper_meta(paper, pdf_path, topic_buckets)
    except Exception as e:
        return False, f"[Failed] {pdf_path}: {e}", None


def download_arxiv_pdfs(
    max_results=1000,
    save_dir="arxiv_pdfs",
    workers=4,
    categories=None,
    include_phrases=None,
    exclude_phrases=None,
    topic_buckets=None,
):
    os.makedirs(save_dir, exist_ok=True)
    categories = categories or DEFAULT_CATEGORIES
    include_phrases = include_phrases or DEFAULT_INCLUDE_PHRASES
    exclude_phrases = exclude_phrases or DEFAULT_EXCLUDE_PHRASES
    topic_buckets = topic_buckets or DEFAULT_TOPIC_BUCKETS

    query = _build_query(categories, include_phrases)
    print(f"[Query] {query}")

    # Use client-level throttling to be friendlier to arXiv.
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    selected = []
    seen_base_id = set()
    for paper in client.results(search):
        # Keep only latest version per base id.
        base_id = re.sub(r"v\d+$", "", paper.get_short_id())
        if base_id in seen_base_id:
            continue
        seen_base_id.add(base_id)
        if _paper_matches_filters(paper, include_phrases, exclude_phrases):
            selected.append(paper)

    print(f"[Selected] {len(selected)} / {len(seen_base_id)} (after filtering)")

    ok = 0
    failed = 0
    metas = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_paper_pdf, p, save_dir, topic_buckets)
            for p in selected
        ]
        for f in as_completed(futures):
            success, msg, meta = f.result()
            print(msg)
            if success:
                ok += 1
                if meta:
                    metas.append(meta)
            else:
                failed += 1

    meta_path = os.path.join(
        save_dir, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(meta_path, "w", encoding="utf-8") as fw:
        for m in metas:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Dataset statistics for reporting in paper/appendix.
    all_category_counter = Counter()
    primary_category_counter = Counter()
    topic_counter = Counter()
    year_counter = Counter()
    for m in metas:
        for c in m.get("categories", []):
            all_category_counter[c] += 1
        if m.get("primary_category"):
            primary_category_counter[m["primary_category"]] += 1
        for t in m.get("topics", []):
            topic_counter[t] += 1
        if m.get("year"):
            year_counter[m["year"]] += 1

    stats = {
        "query": query,
        "max_results": max_results,
        "selected_after_filtering": len(selected),
        "download_success": ok,
        "download_failed": failed,
        "all_category_counts": dict(sorted(all_category_counter.items())),
        "primary_category_counts": dict(sorted(primary_category_counter.items())),
        "topic_bucket_counts": dict(sorted(topic_counter.items())),
        "year_counts": dict(sorted(year_counter.items())),
    }
    stats_path = os.path.join(
        save_dir, f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(stats_path, "w", encoding="utf-8") as fw:
        json.dump(stats, fw, ensure_ascii=False, indent=2)

    print(f"[Done] ok={ok}, failed={failed}, metadata={meta_path}, stats={stats_path}")
    print(f"[Stats] primary categories: {stats['primary_category_counts']}")
    print(f"[Stats] topic buckets: {stats['topic_bucket_counts']}")


if __name__ == "__main__":
    download_arxiv_pdfs(
        max_results=2500,
        workers=4,
    )

import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
DEFAULT_QUALITY_BOOST_PHRASES = [
    "benchmark",
    "dataset",
    "real-world",
    "real robot",
    "sim2real",
    "open-source",
    "github",
    "ablation",
    "evaluation",
    "state of the art",
]
DEFAULT_QUALITY_PENALTY_PHRASES = [
    "position paper",
    "extended abstract",
    "workshop summary",
]


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


def _quality_score(
    paper,
    categories,
    include_phrases,
    boost_phrases,
    penalty_phrases,
):
    title = paper.title or ""
    abstract = paper.summary or ""
    text = f"{title}\n{abstract}".lower()
    score = 0.0

    include_hit = sum(1 for p in include_phrases if p.lower() in text)
    boost_hit = sum(1 for p in boost_phrases if p.lower() in text)
    penalty_hit = sum(1 for p in penalty_phrases if p.lower() in text)
    score += include_hit * 2.0
    score += boost_hit * 1.2
    score -= penalty_hit * 2.0

    # Prefer papers whose primary category is in target robotics-related buckets.
    cats = list(getattr(paper, "categories", []) or [])
    if cats:
        if cats[0] in categories:
            score += 2.5
        score += min(2.0, 0.5 * sum(1 for c in cats if c in categories))

    # Penalize very short abstracts; reward sufficiently detailed abstracts.
    abs_len = len(abstract)
    if abs_len < 300:
        score -= 2.0
    elif abs_len >= 1200:
        score += 1.5
    elif abs_len >= 700:
        score += 1.0

    # Slight preference for concise, informative titles.
    title_words = len(title.split())
    if 5 <= title_words <= 22:
        score += 0.6

    return round(score, 3)


def _build_query(categories, include_phrases, date_range=None):
    cat_clause = " OR ".join([f"cat:{c}" for c in categories])
    kw_clause = " OR ".join([f'ti:"{k}" OR abs:"{k}"' for k in include_phrases])
    query = f"({cat_clause}) AND ({kw_clause})"
    if date_range:
        query += f" AND submittedDate:[{date_range[0]} TO {date_range[1]}]"
    return query


def _year_date_range(year: int):
    # arXiv query time format: YYYYMMDDHHMM
    return (f"{year}01010000", f"{year}12312359")


def _paper_meta(paper, pdf_path):
    return {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": [a.name for a in (paper.authors or [])],
        "published": str(paper.published) if getattr(paper, "published", None) else "",
        "updated": str(paper.updated) if getattr(paper, "updated", None) else "",
        "categories": list(paper.categories or []),
        "pdf_url": paper.pdf_url,
        "pdf_path": pdf_path,
    }


def download_paper_pdf(paper, save_dir):
    paper_id = _safe_paper_id(paper)
    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    # Skip if exists and reasonably large.
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 100 * 1024:
        return True, f"[Skip] {pdf_path}", _paper_meta(paper, pdf_path)

    try:
        paper.download_pdf(filename=pdf_path)
        return True, f"[OK] {pdf_path}", _paper_meta(paper, pdf_path)
    except Exception as e:
        return False, f"[Failed] {pdf_path}: {e}", None


def download_arxiv_pdfs(
    max_results=1000,
    save_dir="arxiv_pdfs_filtered",
    workers=4,
    categories=None,
    include_phrases=None,
    exclude_phrases=None,
    years_back=5,
    per_year_overfetch=3,
    min_quality_score=2.0,
    boost_phrases=None,
    penalty_phrases=None,
):
    os.makedirs(save_dir, exist_ok=True)
    categories = categories or DEFAULT_CATEGORIES
    include_phrases = include_phrases or DEFAULT_INCLUDE_PHRASES
    exclude_phrases = exclude_phrases or DEFAULT_EXCLUDE_PHRASES
    boost_phrases = boost_phrases or DEFAULT_QUALITY_BOOST_PHRASES
    penalty_phrases = penalty_phrases or DEFAULT_QUALITY_PENALTY_PHRASES

    # Use client-level throttling to be friendlier to arXiv.
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    current_year = datetime.now().year
    target_years = [current_year - i for i in range(max(1, years_back))]
    per_year_target = max(1, max_results // len(target_years))
    # Make sure total target is at least max_results.
    remainder = max_results - per_year_target * len(target_years)
    year_targets = {y: per_year_target for y in target_years}
    for y in target_years[: max(0, remainder)]:
        year_targets[y] += 1

    print(
        f"[Plan] years={target_years}, targets={year_targets}, total_target={sum(year_targets.values())}"
    )

    seen_base_id = set()
    scanned = 0
    selected = 0
    ok = 0
    failed = 0
    metas = []
    year_stats = {
        y: {
            "target": year_targets[y],
            "scanned": 0,
            "selected": 0,
            "ok": 0,
            "failed": 0,
            "qualified": 0,
        }
        for y in target_years
    }
    max_inflight = max(1, workers * 4)

    def _download_task(paper, year):
        success, msg, meta = download_paper_pdf(paper, save_dir)
        return year, success, msg, meta

    def _drain_completed(future_set, block=False):
        nonlocal ok, failed
        if not future_set:
            return future_set
        if block:
            done, pending = wait(future_set, return_when=FIRST_COMPLETED)
        else:
            done = {f for f in future_set if f.done()}
            pending = future_set - done
        for f in done:
            year, success, msg, meta = f.result()
            print(msg)
            if success:
                ok += 1
                year_stats[year]["ok"] += 1
                if meta:
                    metas.append(meta)
            else:
                failed += 1
                year_stats[year]["failed"] += 1
        return pending

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = set()
        for year in target_years:
            target = year_targets[year]
            selected_this_year = 0
            date_range = _year_date_range(year)
            query = _build_query(categories, include_phrases, date_range=date_range)
            search = arxiv.Search(
                query=query,
                max_results=target * max(1, per_year_overfetch),
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            print(f"[Year {year}] target={target}, query={query}")
            candidates = []

            for paper in client.results(search):
                scanned += 1
                year_stats[year]["scanned"] += 1

                # Keep only latest version per base id.
                base_id = re.sub(r"v\d+$", "", paper.get_short_id())
                if base_id in seen_base_id:
                    continue
                seen_base_id.add(base_id)

                if not _paper_matches_filters(paper, include_phrases, exclude_phrases):
                    if scanned % 100 == 0:
                        print(
                            f"[Progress] scanned={scanned}, selected={selected}, inflight={len(futures)}, ok={ok}, failed={failed}"
                        )
                    continue

                q = _quality_score(
                    paper,
                    categories=categories,
                    include_phrases=include_phrases,
                    boost_phrases=boost_phrases,
                    penalty_phrases=penalty_phrases,
                )
                if q >= min_quality_score:
                    candidates.append((q, paper))
                    year_stats[year]["qualified"] += 1

                if scanned % 100 == 0:
                    print(
                        f"[Progress] scanned={scanned}, selected={selected}, inflight={len(futures)}, ok={ok}, failed={failed}"
                    )

            candidates.sort(key=lambda x: x[0], reverse=True)
            shortlisted = candidates[:target]
            if candidates:
                best = candidates[0][0]
                worst = candidates[-1][0]
                print(
                    f"[Year {year}] qualified={len(candidates)}, score_range=[{best}, {worst}], min_quality_score={min_quality_score}"
                )
            else:
                print(
                    f"[Year {year}] qualified=0, min_quality_score={min_quality_score}"
                )
            if len(shortlisted) < target:
                print(
                    f"[Year {year}] warning: selected only {len(shortlisted)} / target {target}. Consider lowering min_quality_score or increasing per_year_overfetch."
                )

            for q, paper in shortlisted:
                _ = q
                selected += 1
                selected_this_year += 1
                year_stats[year]["selected"] += 1
                futures.add(executor.submit(_download_task, paper, year))
                if len(futures) >= max_inflight:
                    futures = _drain_completed(futures, block=True)

            print(
                f"[Year {year}] selected={selected_this_year}/{target}, global_selected={selected}"
            )

        while futures:
            futures = _drain_completed(futures, block=True)

    print(f"[Selected] {selected} / {len(seen_base_id)} (after filtering)")
    print("[Yearly Stats]")
    for y in target_years:
        s = year_stats[y]
        print(
            f"  - {y}: target={s['target']}, scanned={s['scanned']}, qualified={s['qualified']}, selected={s['selected']}, ok={s['ok']}, failed={s['failed']}"
        )

    meta_path = os.path.join(
        save_dir, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(meta_path, "w", encoding="utf-8") as fw:
        for m in metas:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[Done] ok={ok}, failed={failed}, metadata={meta_path}")


if __name__ == "__main__":
    download_arxiv_pdfs(
        max_results=5000,
        workers=4,
    )

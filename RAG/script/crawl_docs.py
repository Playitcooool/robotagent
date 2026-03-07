import argparse
import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BLOCKED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".mp4",
    ".mp3",
    ".avi",
}

DEFAULT_ROBOTICS_DOC_SITES = [
    {"url": "https://gazebosim.org/docs/latest/", "depth": 2},
    {"url": "https://docs.ros.org/en/humble/", "depth": 2},
    {"url": "https://navigation.ros.org/", "depth": 2},
    {"url": "https://moveit.picknik.ai/main/doc/", "depth": 2},
    {"url": "https://mujoco.readthedocs.io/en/stable/", "depth": 2},
    {"url": "https://maniskill.readthedocs.io/en/latest/", "depth": 2},
    {"url": "https://pybullet.org/wordpress/", "depth": 2},
]


def clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\-_ .]", "_", name)
    return name[:150]


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.endswith("/"):
        return url[:-1]
    return url


def get_domain_from_url(url: str) -> str:
    return urlparse(url).netloc


def extract_main_content(soup: BeautifulSoup):
    selectors = [
        "main",
        "#content",
        ".content",
        ".documentation",
        ".docs-content",
        ".markdown-body",
        "article",
        ".main-content",
        "#main-content",
        ".doc-content",
    ]
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element
    return soup.find("body") or soup


def _url_to_file_path(save_dir: str, domain: str, title: str, url: str) -> str:
    domain_dir = domain.replace(".", "_")
    sub_dir = os.path.join(save_dir, domain_dir)
    os.makedirs(sub_dir, exist_ok=True)
    title_safe = clean_filename(title or "document")
    short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    filename = f"{title_safe}_{short_hash}.md"
    return os.path.join(sub_dir, filename)


def _iter_internal_links(soup: BeautifulSoup, url: str, domain: str, visited: set[str]) -> list[str]:
    links_to_crawl = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#"):
            continue
        next_url = urljoin(url, href)
        next_url = normalize_url(next_url)
        if not next_url:
            continue
        parsed = urlparse(next_url)
        if parsed.netloc != domain:
            continue
        if any(parsed.path.lower().endswith(ext) for ext in BLOCKED_EXTENSIONS):
            continue
        if next_url in visited:
            continue
        links_to_crawl.append(next_url)
    return sorted(set(links_to_crawl))


def crawl_and_convert(
    url: str,
    domain: str,
    save_dir: str = "RAG/docs_md",
    delay: float = 1.0,
    max_depth: int = 2,
    current_depth: int = 0,
    visited: set[str] | None = None,
    session: requests.Session | None = None,
    min_markdown_chars: int = 200,
):
    if current_depth > max_depth:
        return

    if visited is None:
        visited = set()
    if session is None:
        session = requests.Session()

    url = normalize_url(url)
    if not url or url in visited:
        return
    visited.add(url)
    logger.info("Crawling (%s/%s): %s", current_depth, max_depth, url)

    try:
        time.sleep(max(0.0, delay))
        resp = session.get(url, timeout=12)
        resp.raise_for_status()
        if "text/html" not in (resp.headers.get("Content-Type", "").lower()):
            return
    except Exception as e:
        logger.warning("Fetch failed %s: %s", url, e)
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    main = extract_main_content(soup)
    if not main:
        return

    try:
        md = markdownify(str(main), heading_style="ATX").strip()
    except Exception as e:
        logger.warning("Markdown conversion failed %s: %s", url, e)
        return

    if len(md) < min_markdown_chars:
        logger.info("Skip short page (%s chars): %s", len(md), url)
        return

    title = (soup.title.text.strip() if soup.title else "document").strip()
    filepath = _url_to_file_path(save_dir, domain, title, url)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"Source URL: {url}\n\n")
            f.write("---\n\n")
            f.write(md)
        logger.info("Saved: %s", filepath)
    except Exception as e:
        logger.warning("Write failed %s: %s", filepath, e)
        return

    if current_depth >= max_depth:
        return

    links = _iter_internal_links(soup, url, domain, visited)
    logger.info("Found %s internal links from %s", len(links), url)
    for next_url in links:
        crawl_and_convert(
            next_url,
            domain=domain,
            save_dir=save_dir,
            delay=delay,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            visited=visited,
            session=session,
            min_markdown_chars=min_markdown_chars,
        )


def batch_crawl(
    urls,
    save_dir: str = "RAG/docs_md",
    delay: float = 1.0,
    max_depth: int = 2,
    max_workers: int = 3,
    min_markdown_chars: int = 200,
):
    tasks = []
    for item in urls:
        if isinstance(item, dict):
            url = normalize_url(item.get("url", ""))
            domain = item.get("domain") or get_domain_from_url(url)
            depth = int(item.get("depth", max_depth))
        else:
            url = normalize_url(str(item))
            domain = get_domain_from_url(url)
            depth = max_depth
        if url and domain:
            tasks.append((url, domain, depth))

    logger.info("Start crawling %s sites", len(tasks))
    os.makedirs(save_dir, exist_ok=True)

    def _run_site(root_url: str, domain: str, depth: int):
        visited = set()
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "RobotAgent-RAG-DocCrawler/1.0 (+https://localhost)",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        crawl_and_convert(
            root_url,
            domain=domain,
            save_dir=save_dir,
            delay=delay,
            max_depth=depth,
            visited=visited,
            session=session,
            min_markdown_chars=min_markdown_chars,
        )
        return {"site": root_url, "visited_pages": len(visited)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_site, u, d, dep) for (u, d, dep) in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
                logger.info("Done site: %s", result)
            except Exception as e:
                logger.error("Site crawl failed: %s", e)


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl docs/blog pages and convert to markdown.")
    parser.add_argument("--save-dir", default="RAG/docs_md")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--min-markdown-chars", type=int, default=200)
    parser.add_argument(
        "--preset",
        action="store_true",
        help="Use built-in robotics doc sites list.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Add one seed URL. Repeatable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = []
    if args.preset:
        seeds.extend(DEFAULT_ROBOTICS_DOC_SITES)
    if args.url:
        seeds.extend(args.url)
    if not seeds:
        seeds = DEFAULT_ROBOTICS_DOC_SITES

    batch_crawl(
        seeds,
        save_dir=args.save_dir,
        delay=args.delay,
        max_depth=args.max_depth,
        max_workers=args.max_workers,
        min_markdown_chars=args.min_markdown_chars,
    )

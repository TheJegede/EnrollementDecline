"""Phase 1 — fetch NCES tables, IPEDS surveys, ASU corpus.

Functions are idempotent: skip files that already exist on disk.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
import urllib.robotparser
from pathlib import Path
from typing import Iterable

import requests

from src.utils import CORPUS_DIR, IPEDS_DIR, NCES_DIR

log = logging.getLogger(__name__)

USER_AGENT = (
    "EnrollmentAI-PortfolioPOC/0.1 "
    "(+https://github.com/your-handle ; contact: jegedetaiwo95@gmail.com)"
)

NCES_TABLES = {
    # 219.10: high school graduates by state — d22
    "tabn219.10.xls": "https://nces.ed.gov/programs/digest/d22/tables/xls/tabn219.10.xls",
    # 303.10: enrollment projections — only published in d21 latest
    "tabn303.10.xls": "https://nces.ed.gov/programs/digest/d21/tables/xls/tabn303.10.xls",
    # 302.10: enrollment by age — d22
    "tabn302.10.xls": "https://nces.ed.gov/programs/digest/d22/tables/xls/tabn302.10.xls",
}

# IPEDS file naming: ADM/IC are single zips per year; EF is split into subparts.
# EFFY = 12-month by race/ethnicity (best for demographic enrollment analysis).
IPEDS_FILES_TEMPLATE = ["ADM{YEAR}", "IC{YEAR}", "EFFY{YEAR}"]
IPEDS_YEARS = [2018, 2019, 2020, 2021, 2022]


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (exists): %s", dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info("downloaded: %s (%.1f KB)", dest.name, len(resp.content) / 1024)
        return True
    except requests.RequestException as e:
        log.warning("failed %s -> %s: %s", url, dest.name, e)
        return False


def fetch_nces_tables(target_dir: Path = NCES_DIR) -> dict[str, bool]:
    target_dir.mkdir(parents=True, exist_ok=True)
    return {name: _download(url, target_dir / name) for name, url in NCES_TABLES.items()}


def fetch_ipeds(
    target_dir: Path = IPEDS_DIR,
    file_templates: Iterable[str] = IPEDS_FILES_TEMPLATE,
    years: Iterable[int] = IPEDS_YEARS,
) -> dict[str, bool]:
    """Fetch IPEDS complete-data zips."""
    target_dir.mkdir(parents=True, exist_ok=True)
    base = "https://nces.ed.gov/ipeds/datacenter/data"
    results: dict[str, bool] = {}
    for tmpl in file_templates:
        for year in years:
            name = tmpl.replace("{YEAR}", str(year)) + ".zip"
            url = f"{base}/{name}"
            results[name] = _download(url, target_dir / name)
    return results


_robots_cache: dict[str, list[str]] = {}


def _load_disallow_rules(host_root: str) -> list[str]:
    """Parse Disallow paths under User-agent: * from a robots.txt.

    Python's stdlib RobotFileParser has long-standing bugs with Allow lines that
    use $ anchors (returns False for everything) — bypass it with a minimal
    Disallow-only parser. Path-prefix matching is sufficient for our crawl.
    """
    if host_root in _robots_cache:
        return _robots_cache[host_root]
    try:
        resp = requests.get(
            f"{host_root}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=15
        )
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as e:
        log.warning("robots.txt read failed for %s: %s", host_root, e)
        _robots_cache[host_root] = []
        return []

    disallow: list[str] = []
    in_star = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            in_star = value == "*"
        elif key == "disallow" and in_star and value:
            disallow.append(value)
    _robots_cache[host_root] = disallow
    return disallow


def _robots_allows(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host_root = f"{parsed.scheme}://{parsed.netloc}"
    rules = _load_disallow_rules(host_root)
    path = parsed.path or "/"
    return not any(path.startswith(rule) for rule in rules)


_NOISE_PATH_FRAGMENTS = (
    "/cdn-cgi/",
    "/sites/default/files/",
    "/print/",
    "/rss/",
    "/feed",
)
_ALLOWED_DOMAINS = {
    "admission.asu.edu",
    "students.asu.edu",
    "tuition.asu.edu",
    "asu.edu",
}


def _normalize_url(url: str) -> str:
    """Strip fragment and trailing slash so /x and /x#frag dedupe."""
    parsed = urllib.parse.urlparse(url)
    cleaned = parsed._replace(fragment="")
    return urllib.parse.urlunparse(cleaned)


def _crawlable(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in _ALLOWED_DOMAINS:
        return False
    if any(fragment in parsed.path for fragment in _NOISE_PATH_FRAGMENTS):
        return False
    return True


def scrape_asu_corpus(
    seed_urls: list[str],
    out_dir: Path = CORPUS_DIR,
    max_pages: int = 300,
    throttle_seconds: float = 1.0,
) -> int:
    """Crawl ASU admissions, BFS through same-domain links. Returns files written."""
    import trafilatura
    from bs4 import BeautifulSoup

    out_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    queue: list[str] = [_normalize_url(u) for u in seed_urls]
    written = 0

    while queue and written < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if not _crawlable(url):
            continue
        if not _robots_allows(url):
            log.info("robots disallow: %s", url)
            continue

        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("fetch failed %s: %s", url, e)
            continue

        markdown = trafilatura.extract(
            resp.text, output_format="markdown", include_links=False
        )
        if markdown and len(markdown) > 200:
            slug = urllib.parse.quote(url, safe="")[:200]
            (out_dir / f"{slug}.md").write_text(
                f"# Source: {url}\n\n{markdown}\n", encoding="utf-8"
            )
            written += 1

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = _normalize_url(urllib.parse.urljoin(url, a["href"]))
            if href and href not in visited and _crawlable(href):
                queue.append(href)

        time.sleep(throttle_seconds)

    log.info("scrape done: %d files in %s", written, out_dir)
    return written


ASU_SEED_URLS = [
    "https://admission.asu.edu/",
    "https://admission.asu.edu/freshman",
    "https://admission.asu.edu/transfer",
    "https://admission.asu.edu/international",
    "https://admission.asu.edu/aid",
    "https://admission.asu.edu/apply",
    "https://admission.asu.edu/contact",
    "https://admission.asu.edu/programs",
    "https://admission.asu.edu/visit",
    "https://admission.asu.edu/admitted",
    "https://students.asu.edu/costs",
    "https://students.asu.edu/financialaid",
    "https://students.asu.edu/scholarships",
    "https://students.asu.edu/housing",
    "https://tuition.asu.edu/",
]

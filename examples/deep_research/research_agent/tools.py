"""Research tools.

Tavily is used for URL discovery. Candidate pages are then validated and
downloaded locally with HTTPX.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from typing_extensions import Annotated, Literal


tavily_client = TavilyClient()

# One page should not consume most of the model context.
MAX_CONTENT_CHARS = 16_000

# Stop after obtaining this many readable sources from one search.
MAX_SUCCESSFUL_SOURCES = 2

# Do not repeatedly wait for the same failed URL within nearby runs.
FAILED_URL_TTL_SECONDS = 600.0

# Known obsolete deployment domains. Do not spend timeout budget on them.
BLOCKED_LEGACY_HOSTS = {
    "sgl-project-sglang-93.mintlify.app",
}

_failed_urls: dict[str, float] = {}
_failed_urls_lock = Lock()


@dataclass(frozen=True)
class FetchResult:
    """Result of validating and downloading one webpage."""

    requested_url: str
    final_url: str | None
    content: str | None
    error: str | None


def _canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication and failure caching."""
    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()

    if parts.port:
        netloc = f"{hostname}:{parts.port}"
    else:
        netloc = hostname

    path = parts.path or "/"

    # Fragments do not change the downloaded document.
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _source_priority(url: str) -> int:
    """Prefer likely primary-source domains over blogs."""
    host = _hostname(url)

    if (
        host == "arxiv.org"
        or host.endswith(".gov")
        or host.endswith(".edu")
        or host.startswith("docs.")
        or host == "github.com"
    ):
        return 0

    if host.endswith("readthedocs.io"):
        return 1

    return 2


def _recently_failed(url: str) -> bool:
    """Return True when this URL failed during the recent TTL window."""
    now = time.monotonic()

    with _failed_urls_lock:
        expired = [
            failed_url
            for failed_url, failed_at in _failed_urls.items()
            if now - failed_at >= FAILED_URL_TTL_SECONDS
        ]

        for failed_url in expired:
            del _failed_urls[failed_url]

        return url in _failed_urls


def _remember_failure(url: str) -> None:
    with _failed_urls_lock:
        _failed_urls[url] = time.monotonic()


def fetch_webpage_content(
    url: str,
    timeout: float = 15.0,
) -> FetchResult:
    """Fetch one webpage, following redirects and limiting content size."""
    requested_url = _canonicalize_url(url)

    if urlsplit(requested_url).scheme not in {"http", "https"}:
        return FetchResult(
            requested_url=requested_url,
            final_url=None,
            content=None,
            error="Unsupported URL scheme",
        )

    host = _hostname(requested_url)

    if host in BLOCKED_LEGACY_HOSTS:
        return FetchResult(
            requested_url=requested_url,
            final_url=None,
            content=None,
            error=f"Skipped known legacy host: {host}",
        )

    if _recently_failed(requested_url):
        return FetchResult(
            requested_url=requested_url,
            final_url=None,
            content=None,
            error="Skipped because this URL failed recently",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    }

    try:
        response = httpx.get(
            requested_url,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        response.raise_for_status()

        final_url = _canonicalize_url(str(response.url))
        content_type = response.headers.get("content-type", "").lower()

        supported_content = (
            content_type.startswith("text/")
            or "application/xhtml+xml" in content_type
            or "application/json" in content_type
        )

        if not supported_content:
            error = f"Unsupported content type: {content_type or 'unknown'}"
            _remember_failure(requested_url)
            return FetchResult(
                requested_url=requested_url,
                final_url=final_url,
                content=None,
                error=error,
            )

        if "html" in content_type:
            content = markdownify(response.text)
        else:
            content = response.text

        content = content.strip()

        if not content:
            _remember_failure(requested_url)
            return FetchResult(
                requested_url=requested_url,
                final_url=final_url,
                content=None,
                error="Downloaded page was empty",
            )

        if len(content) > MAX_CONTENT_CHARS:
            content = (
                content[:MAX_CONTENT_CHARS]
                + "\n\n[Content truncated by the research tool]"
            )

        return FetchResult(
            requested_url=requested_url,
            final_url=final_url,
            content=content,
            error=None,
        )

    except Exception as exc:
        _remember_failure(requested_url)

        return FetchResult(
            requested_url=requested_url,
            final_url=None,
            content=None,
            error=f"{type(exc).__name__}: {exc}",
        )


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"],
        InjectedToolArg,
    ] = "general",
) -> str:
    """Search for live webpages and return readable source content.

    Use include_domains when the user requests primary or official sources.
    Failed and duplicate URLs are skipped, and another candidate is tried.

    Args:
        query: Search query to execute.
        include_domains: Trusted domains to prioritize or restrict results to.
        exclude_domains: Domains that must not appear in the results.
        max_results: Maximum candidate URLs requested from Tavily.
        topic: Tavily topic filter.

    Returns:
        Readable content from successful sources plus a concise failure summary.
    """
    search_kwargs: dict[str, object] = {
        "query": query,
        "max_results": max_results,
        "topic": topic,
    }

    if include_domains:
        search_kwargs["include_domains"] = include_domains

    if exclude_domains:
        search_kwargs["exclude_domains"] = exclude_domains

    search_results = tavily_client.search(**search_kwargs)
    candidates = search_results.get("results", [])

    # Preserve Tavily relevance within each source-priority tier.
    candidates = sorted(
        enumerate(candidates),
        key=lambda item: (
            _source_priority(item[1].get("url", "")),
            item[0],
        ),
    )

    successful_results: list[str] = []
    failures: list[str] = []
    seen_urls: set[str] = set()

    for _, result in candidates:
        if len(successful_results) >= MAX_SUCCESSFUL_SOURCES:
            break

        raw_url = result.get("url", "")
        title = result.get("title", "Untitled source")
        url = _canonicalize_url(raw_url)

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        fetched = fetch_webpage_content(url)

        if fetched.error is not None:
            failures.append(f"- {url}: {fetched.error}")
            continue

        final_url = fetched.final_url or url

        successful_results.append(
            "\n".join(
                [
                    f"## {title}",
                    f"**URL:** {final_url}",
                    "",
                    fetched.content or "",
                    "",
                    "---",
                ]
            )
        )

    sections = [
        (
            f"Search query: {query}\n"
            f"Candidates returned: {len(candidates)}\n"
            f"Readable sources: {len(successful_results)}"
        )
    ]

    if successful_results:
        sections.append("\n\n".join(successful_results))
    else:
        sections.append(
            "No candidate page could be downloaded. "
            "Do not infer factual claims from this search."
        )

    if failures:
        sections.append(
            "Skipped or failed candidates:\n"
            + "\n".join(failures[:5])
        )

    return "\n\n".join(sections)




@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

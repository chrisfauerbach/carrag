from datetime import datetime, timezone

import httpx
import trafilatura
from bs4 import BeautifulSoup


async def parse_url(url: str) -> dict:
    """Fetch and extract clean text from a web page.

    Uses trafilatura with BeautifulSoup fallback.
    Returns dict with 'content' and 'metadata'.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Try trafilatura first
    content = trafilatura.extract(html, include_comments=False, include_tables=True)

    title = None
    if content is None:
        # Fallback to BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else None
    else:
        # Extract title from HTML even when trafilatura succeeds
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title else None

    metadata = {
        "filename": url,
        "source_type": "web",
        "url": url,
        "title": title,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"content": content or "", "metadata": metadata}

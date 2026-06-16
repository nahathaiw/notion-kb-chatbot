"""
Pull documents straight from a real Notion workspace via the Notion API.

WHY THIS EXISTS
The base demo reads `.txt` files from `docs/`. That's great for learning, but a
"real" knowledge-base bot should answer from your ACTUAL Notion pages. This
module does exactly that: it asks Notion for every page your integration can
see, walks each page's block tree to extract the plain text, and returns records
in the SAME shape the rest of the pipeline already understands:

    {"source": "<page title>", "text": "<full page text>", "url": "<page url>"}

Because the shape matches `load_documents()`, the chunk -> embed -> retrieve ->
generate pipeline downstream doesn't change at all.

SETUP (one time)
  1. Create an internal integration: https://www.notion.so/my-integrations
     -> "New integration" -> copy the "Internal Integration Secret".
  2. Put it in your .env:   NOTION_API_KEY=secret_xxx   (or NOTION_TOKEN=...)
  3. Share pages with it: open a Notion page -> "..." menu -> "Connections" ->
     add your integration. The API only sees pages you've explicitly shared.

The API is deliberately accessed with plain `requests` so every call is visible,
matching the from-scratch spirit of the rest of this project.
"""

import os
import time

import requests

NOTION_API_BASE = "https://api.notion.com/v1"
# Notion requires pinning an API version via this header. This is a stable,
# widely-supported version; bump it only if you need newer block types.
NOTION_VERSION = "2022-06-28"

# Block types whose payload carries a `rich_text` array we can turn into text.
# (Notion stores the words of a paragraph, heading, list item, etc. as a list of
# "rich text" fragments under the block's own type key.)
_RICH_TEXT_BLOCKS = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do", "toggle",
    "quote", "callout", "code",
}


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _get_token(token=None):
    """Resolve the integration token from the argument or the environment."""
    token = token or os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError(
            "No Notion token found. Set NOTION_API_KEY in your .env "
            "(see notion_loader.py docstring for setup steps)."
        )
    return token


def _rich_text_to_plain(rich_text):
    """Join a Notion `rich_text` array into a single plain string."""
    return "".join(fragment.get("plain_text", "") for fragment in rich_text)


def search_pages(token, page_size=100):
    """Return every page object the integration can access.

    Uses the /search endpoint filtered to pages, following pagination until
    Notion says there are no more results.
    """
    pages = []
    payload = {
        "filter": {"property": "object", "value": "page"},
        "page_size": page_size,
    }
    cursor = None
    while True:
        if cursor:
            payload["start_cursor"] = cursor
        resp = requests.post(
            f"{NOTION_API_BASE}/search", headers=_headers(token), json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return pages


def page_title(page):
    """Best-effort page title from a page object's properties."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title = _rich_text_to_plain(prop.get("title", []))
            if title.strip():
                return title.strip()
    return "Untitled"


def _block_text(block):
    """Extract this block's own text (not its children)."""
    btype = block.get("type")
    if btype not in _RICH_TEXT_BLOCKS:
        return ""
    rich = block.get(btype, {}).get("rich_text", [])
    text = _rich_text_to_plain(rich)
    # Render checkboxes so to-do items read sensibly in the context block.
    if btype == "to_do":
        checked = block.get("to_do", {}).get("checked", False)
        text = f"[{'x' if checked else ' '}] {text}"
    return text


def fetch_block_children_text(block_id, token, depth=0, max_depth=6):
    """Recursively collect text from a block's children.

    Notion stores page content as a tree of blocks; nested blocks (e.g. items
    inside a toggle) are children of their parent. We walk that tree depth-first,
    pulling the text out of each block, with a depth cap to avoid pathological
    nesting.
    """
    if depth > max_depth:
        return []

    lines = []
    cursor = None
    while True:
        params = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        resp = requests.get(
            f"{NOTION_API_BASE}/blocks/{block_id}/children",
            headers=_headers(token), params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for block in data.get("results", []):
            text = _block_text(block)
            if text:
                lines.append(("  " * depth) + text)
            if block.get("has_children"):
                lines.extend(
                    fetch_block_children_text(block["id"], token, depth + 1, max_depth)
                )

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        # Be gentle with Notion's ~3 req/s rate limit on big workspaces.
        time.sleep(0.2)

    return lines


def fetch_notion_documents(token=None, progress=None):
    """Fetch all accessible Notion pages as {"source", "text", "url"} records.

    `progress` is an optional callback(done, total, title) for UI feedback.
    Pages with no extractable text are skipped (e.g. empty pages or pure
    database rows we couldn't read).
    """
    token = _get_token(token)
    pages = search_pages(token)

    documents = []
    for i, page in enumerate(pages):
        title = page_title(page)
        if progress:
            progress(i, len(pages), title)
        lines = fetch_block_children_text(page["id"], token)
        text = "\n".join(lines).strip()
        if not text:
            continue
        documents.append({
            "source": title,
            "text": text,
            "url": page.get("url", ""),
        })

    if not documents:
        raise RuntimeError(
            "No readable Notion pages found. Make sure you've shared at least one "
            "page with your integration (page -> ... -> Connections)."
        )
    return documents

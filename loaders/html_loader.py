"""
loaders/html_loader.py
----------------------
Extracts readable text from HTML pages, grouped by heading.

Web pages contain a lot that is NOT knowledge: scripts, styles, navigation
menus, cookie banners, footers. Embedding that noise would pollute the
vector store, so this loader:

    1. removes <script>, <style>, <nav>, <footer>, <header> nav bars, forms,
       and hidden elements outright;
    2. walks the remaining body in document order and starts a new
       `LoadedDocument` at every <h1>/<h2>/<h3>, recording the heading path
       in `section` (e.g. "Our Offices > Dubai");
    3. records the page <title> in metadata so citations can show it.

Library: BeautifulSoup with the fast lxml parser.
"""

from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from .base import LoadedDocument, base_metadata, normalize_whitespace

# Elements that never contain knowledge-base content.
_NOISE_TAGS = ("script", "style", "noscript", "nav", "footer", "form", "iframe", "svg", "button")
_HEADING_TAGS = ("h1", "h2", "h3")
_BLOCK_TAGS = ("p", "li", "td", "th", "pre", "blockquote", "div", "section", "article", "span")


def load_html(path: Path) -> list[LoadedDocument]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    page_title = soup.title.get_text(strip=True) if soup.title else path.stem

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()  # remove noise completely, including its text

    body = soup.body or soup
    documents: list[LoadedDocument] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush():
        if buffer and buffer[0].strip() == (heading_stack[-1] if heading_stack else '').strip() and not any(l.strip() for l in buffer[1:]):
            return  # heading with no body yet: merge it into the next section
        text = normalize_whitespace("\n".join(buffer))
        if text:
            metadata = base_metadata(path)
            metadata["title"] = page_title
            metadata["section"] = " > ".join(h for h in heading_stack if h) or "Preamble"
            documents.append(LoadedDocument(text=text, metadata=metadata))
        buffer.clear()

    def walk(node):
        """Depth-first traversal that emits text in reading order."""
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    buffer.append(text)
            elif isinstance(child, Tag):
                if child.name in _HEADING_TAGS:
                    flush()
                    level = int(child.name[1])
                    title = child.get_text(" ", strip=True)
                    del heading_stack[level - 1:]
                    heading_stack.extend([""] * (level - 1 - len(heading_stack)))
                    heading_stack.append(title)
                    buffer.append(title)
                elif child.name == "li":
                    # Render list items on their own line with a bullet.
                    buffer.append("- " + child.get_text(" ", strip=True))
                elif child.name in ("p", "td", "th", "pre", "blockquote"):
                    # Leaf-ish blocks: take their whole text in one go so
                    # inline tags (<strong>, <a>) do not fragment sentences.
                    text = child.get_text(" ", strip=True)
                    if text:
                        buffer.append(text)
                else:
                    walk(child)  # containers: keep descending

    walk(body)
    flush()
    return documents

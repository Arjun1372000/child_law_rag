"""Document loading and statute-aware chunking for the Child Law RAG corpus."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

from config import DATA_DIR, CENTRAL_ACTS_MANIFEST, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def is_constitution_page_relevant(text: str) -> bool:
    t = text.lower()
    keywords = [
        "child", "children", "minor", "21a", "article 21a", "article 24",
        "article 39", "article 45", "article 15(3)", "right to education",
        "compulsory education", "tender age", "trafficking"
    ]
    return any(k in t for k in keywords)


def load_documents(folder: Optional[Path] = None) -> List[Dict]:
    base_path = Path(folder) if folder else DATA_DIR
    documents: List[Dict] = []

    for filename, act_title in CENTRAL_ACTS_MANIFEST.items():
        file_path = base_path / filename
        if not file_path.exists():
            print(f"WARNING: required corpus file missing: {file_path}")
            continue

        try:
            reader = PdfReader(str(file_path))
            page_texts = []
            for page in reader.pages:
                raw = page.extract_text() or ""
                if filename == "constitution_english.pdf" and not is_constitution_page_relevant(raw):
                    continue
                page_texts.append(raw)

            text = "\n\n".join(page_texts)
            text = re.sub(r"\r\n?|\r", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            documents.append({"file": filename, "act_title": act_title, "text": text})
            print(f"Loaded: {filename} | {len(text):,} chars")
        except Exception as exc:
            print(f"ERROR loading {filename}: {exc}")

    return documents


def extract_section_hint(text: str) -> str:
    """Extract an actual legal provision number from a section/chunk header."""
    patterns = [
        r"\bSection\s+(\d+[A-Z]?(?:\(\s*\d+[A-Z]?\s*\))?)\b",
        r"\bArticle\s+(\d+[A-Z]?(?:\([a-z]\))?)\b",
        r"\bRule\s+(\d+[A-Z]?(?:\(\s*\d+\s*\))?)\b",
        r"^\s*(\d+[A-Z]?)\.\s+(?=\(?\d|[A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            if "Section" in pattern:
                return f"Section {match.group(1)}"
            if "Article" in pattern:
                return f"Article {match.group(1)}"
            if "Rule" in pattern:
                return f"Rule {match.group(1)}"
            return f"Section {match.group(1)}"
    return "General Provision"


def _split_statutory_blocks(text: str) -> List[str]:
    """
    Split on real top-level provisions rather than occurrences such as 'section 14'
    inside another provision. PDF extracts commonly represent headings as '15. (1)'.
    """
    boundary = re.compile(
        r"(?=^\s*(?:Section\s+\d+[A-Z]?|Article\s+\d+[A-Z]?|Rule\s+\d+[A-Z]?|\d+[A-Z]?\.\s+\(?\d*|CHAPTER\s+[IVXLCDM\d]+)\b)",
        re.IGNORECASE | re.MULTILINE,
    )
    blocks = [b.strip() for b in boundary.split(text) if b.strip()]
    return blocks or [text.strip()]


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    chunks: List[Dict] = []
    chunk_id = 0

    for doc in documents:
        for block in _split_statutory_blocks(doc["text"]):
            hint = extract_section_hint(block[:500])
            if hint == "General Provision":
                # Check the full block only as a fallback. This is useful for PDFs
                # where a provision heading lands just after a page header.
                hint = extract_section_hint(block)

            if len(block) <= chunk_size + overlap:
                chunks.append({
                    "file": doc["file"],
                    "chunk_id": chunk_id,
                    "act_title": doc["act_title"],
                    "section_hint": hint,
                    "text": block,
                })
                chunk_id += 1
                continue

            start = 0
            while start < len(block):
                end = min(start + chunk_size, len(block))
                if end < len(block):
                    cut = block.rfind("\n", start, end)
                    if cut < start + chunk_size // 2:
                        cut = block.rfind(". ", start, end)
                    if cut > start + chunk_size // 2:
                        end = cut + 1

                piece = block[start:end].strip()
                if len(piece) >= 60:
                    prefix = f"[{doc['act_title']} - {hint}]\n"
                    if hint != "General Provision" and hint.lower() not in piece[:120].lower():
                        piece = prefix + piece
                    chunks.append({
                        "file": doc["file"],
                        "chunk_id": chunk_id,
                        "act_title": doc["act_title"],
                        "section_hint": hint,
                        "text": piece,
                    })
                    chunk_id += 1

                if end >= len(block):
                    break
                start = max(start + 1, end - overlap)

    print(f"Generated {len(chunks)} statutory chunks")
    return chunks

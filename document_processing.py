# document_processing.py
"""
Document loading, statutory preprocessing, and structure-aware chunking
for Central Indian Child Protection Laws.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from config import DATA_DIR, CENTRAL_ACTS_MANIFEST, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def is_constitution_page_relevant(text: str) -> bool:
    """Check if a page of the Indian Constitution relates to child rights or fundamental provisions."""
    t = text.lower()
    keywords = [
        "child", "children", "minor", "tender age", "21a", "article 21a",
        "right to education", "article 24", "article 39", "article 45",
        "article 15(3)", "article 51a", "trafficking", "compulsory education"
    ]
    return any(k in t for k in keywords)


def load_documents(folder: Optional[Path] = None) -> List[Dict]:
    """
    Load Central Indian Child Law documents with statutory filtering.
    
    Args:
        folder: Path to documents folder (defaults to DATA_DIR)
        
    Returns:
        List of document dictionaries containing file, act_title, and text.
    """
    base_path = Path(folder) if folder else DATA_DIR
    documents = []

    for filename, act_title in CENTRAL_ACTS_MANIFEST.items():
        file_path = base_path / filename
        if not file_path.exists():
            print(f"Notice: {filename} not found in {base_path}, skipping.")
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                page_texts = []
                
                for idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    
                    # For the massive 400+ page Constitution, extract only child-relevant articles
                    if filename == "constitution_english.pdf":
                        if is_constitution_page_relevant(raw_text):
                            page_texts.append(raw_text)
                    else:
                        page_texts.append(raw_text)
                        
                text = "\n\n".join(page_texts)
            else:
                text = file_path.read_text(encoding="utf-8")

            # Clean general whitespace while preserving paragraph boundaries
            text = re.sub(r"\r\n|\r", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            documents.append({
                "file": filename,
                "act_title": act_title,
                "text": text
            })
            print(f"Loaded: {filename} ({act_title}) - {len(text)} chars")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


def extract_section_hint(text: str) -> str:
    """Extract section or article identifier from statutory text snippet."""
    patterns = [
        r"(Section\s+\d+[A-Z]?)",
        r"(Article\s+\d+[A-Z]?)",
        r"(Rule\s+\d+[A-Z]?)",
        r"(CHAPTER\s+[IVXLCDM\d]+)",
        r"(Section\s+[0-9]+)",
        r"(Sec\.\s*[0-9]+)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).title()
    return "General Provision"


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """
    Structure-aware legal chunking:
    Preserves statutory sections, articles, and clause boundaries.
    
    Args:
        documents: List of document dicts
        chunk_size: Target size per chunk in characters
        overlap: Character overlap for split sections
        
    Returns:
        List of chunk dicts with file, chunk_id, act_title, section_hint, and text.
    """
    chunks = []
    chunk_id = 0

    # Pattern to split on statutory boundaries (e.g., Section 12, Article 21A, Rule 5)
    boundary_pattern = re.compile(
        r"(?=(?:\n\s*(?:Section\s+\d+|Article\s+\d+|Rule\s+\d+|CHAPTER\s+[IVXLCDM]+|[0-9]+\.\s+[A-Z])))",
        re.IGNORECASE
    )

    for doc in documents:
        text = doc["text"]
        raw_sections = boundary_pattern.split(text)
        
        current_section_hint = "Preamble / Preliminary"

        for section_block in raw_sections:
            section_block = section_block.strip()
            if not section_block:
                continue

            # Update current section context if a new header is encountered
            hint = extract_section_hint(section_block[:200])
            if hint != "General Provision":
                current_section_hint = hint

            # If the statutory section fits reasonably within chunk size, keep it intact
            if len(section_block) <= chunk_size + overlap:
                chunks.append({
                    "file": doc["file"],
                    "chunk_id": chunk_id,
                    "act_title": doc["act_title"],
                    "section_hint": current_section_hint,
                    "text": section_block
                })
                chunk_id += 1
            else:
                # Section is long: split across paragraph breaks with overlap
                start = 0
                while start < len(section_block):
                    end = start + chunk_size
                    # Try to break at a paragraph or sentence boundary
                    if end < len(section_block):
                        break_pos = section_block.rfind("\n", start, end)
                        if break_pos == -1 or break_pos < start + (chunk_size // 2):
                            break_pos = section_block.rfind(". ", start, end)
                        if break_pos != -1 and break_pos > start + (chunk_size // 2):
                            end = break_pos + 1
                    
                    chunk_text = section_block[start:end].strip()
                    if len(chunk_text) > 40:  # Avoid empty or trivial stubs
                        # Prefix continuation with section hint if not already present
                        prefix = f"[{doc['act_title']} - {current_section_hint}]\n"
                        final_text = chunk_text if current_section_hint in chunk_text[:80] else prefix + chunk_text
                        
                        chunks.append({
                            "file": doc["file"],
                            "chunk_id": chunk_id,
                            "act_title": doc["act_title"],
                            "section_hint": current_section_hint,
                            "text": final_text
                        })
                        chunk_id += 1

                    start += max(1, chunk_size - overlap)

    print(f"Total structured statutory chunks generated: {len(chunks)}")
    return chunks

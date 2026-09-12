# document_processing.py
"""
Functions for loading and chunking documents
"""

import re
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict
from config import DATA_DIR


def load_documents(folder=None, include_states=True) -> List[Dict]:
    """
    Load documents from a folder with optional state-specific support
    
    Args:
        folder: Path to the folder containing documents (defaults to DATA_DIR)
        include_states: Whether to load state-specific documents
        
    Returns:
        List of document dictionaries with file, state, and text
    """
    documents = []
    
    # Resolve document directory
    if folder is None:
        base_path = Path(DATA_DIR)
    else:
        base_path = Path(folder)
        if not base_path.exists() and not base_path.is_absolute():
            candidate = DATA_DIR.parent / folder
            if candidate.exists():
                base_path = candidate

    if base_path.exists():
        for file in base_path.glob("*"):
            if file.is_file():
                try:
                    if file.suffix.lower() == ".pdf":
                        reader = PdfReader(str(file))
                        text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                    else:
                        text = file.read_text(encoding="utf-8")

                    documents.append({
                        "file": file.name,
                        "state": None,
                        "text": text
                    })

                except Exception as e:
                    print(f"Could not load {file.name}: {e}")
    
    # Load state-specific documents
    if include_states:
        for state_folder in ["tamil_nadu", "kerala"]:
            state_path = base_path / state_folder
            if state_path.exists():
                for file in state_path.glob("*"):
                    if file.is_file():
                        try:
                            if file.suffix.lower() == ".pdf":
                                reader = PdfReader(str(file))
                                text = "\n".join(
                                    page.extract_text() or "" for page in reader.pages
                                )
                            else:
                                text = file.read_text(encoding="utf-8")

                            documents.append({
                                "file": file.name,
                                "state": state_folder,
                                "text": text
                            })

                        except Exception as e:
                            print(f"Could not load {file.name} from {state_folder}: {e}")
    
    return documents


def chunk_documents(documents: List[Dict], chunk_size: int = 700, overlap: int = 150) -> List[Dict]:
    """
    Split documents into overlapping chunks
    
    Args:
        documents: List of document dictionaries
        chunk_size: Size of each chunk in characters
        overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunk dictionaries with file, chunk_id, and text
    """
    chunks = []

    for doc in documents:
        text = re.sub(r"\s+", " ", doc["text"]).strip()

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "file": doc["file"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            chunk_id += 1
            start += chunk_size - overlap

    return chunks

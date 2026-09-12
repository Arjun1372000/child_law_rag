# embeddings.py
"""
Functions for creating and managing embeddings
"""

import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from config import client, EMBEDDING_MODEL, CACHE_FILE, CACHE_DIR


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Create embeddings for all chunks using the embedding model
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with embedding vectors added
    """
    for chunk in chunks:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk["text"]
        )

        chunk["embedding"] = np.array(response.data[0].embedding)

    return chunks


def save_embeddings_cache(chunks: List[Dict], cache_file: str = CACHE_FILE) -> None:
    """
    Save chunks with embeddings to disk for later use
    
    Args:
        chunks: List of chunk dictionaries with embeddings
        cache_file: Name of the cache file
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    
    cache_path = cache_dir / cache_file
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Saved {len(chunks)} chunks to {cache_path}")


def load_embeddings_cache(cache_file: str = CACHE_FILE) -> List[Dict]:
    """
    Load chunks with embeddings from disk
    
    Args:
        cache_file: Name of the cache file
        
    Returns:
        List of chunks with embeddings, or None if cache doesn't exist
    """
    cache_path = Path(CACHE_DIR) / cache_file
    
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from cache")
        return chunks
    
    return None


def embeddings_cache_exists(cache_file: str = CACHE_FILE) -> bool:
    """
    Check if embeddings cache exists
    
    Args:
        cache_file: Name of the cache file
        
    Returns:
        True if cache file exists, False otherwise
    """
    return (Path(CACHE_DIR) / cache_file).exists()

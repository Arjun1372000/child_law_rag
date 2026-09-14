# embeddings.py
"""
Batch embedding generation with persistent disk checkpointing,
automatic quota resumption (429 handling), and final disk cache.
"""

import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from config import (
    call_embed,
    CACHE_FILE,
    OUTPUTS_DIR,
    BATCH_EMBED_SIZE,
    REQUEST_DELAY_SECONDS
)

CHECKPOINT_FILE = OUTPUTS_DIR / "gemini_embeddings_checkpoint.pkl"


def embed_chunks(chunks: List[Dict], batch_size: int = BATCH_EMBED_SIZE) -> List[Dict]:
    """
    Create embeddings for all chunks with checkpointing and automatic rate-limit recovery.
    """
    total = len(chunks)
    print(f"Embedding {total} chunks using Gemini (batch size: {batch_size})...")

    # Resume from checkpoint if available
    embedded_chunks = []
    start_idx = 0
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                embedded_chunks = pickle.load(f)
            start_idx = len(embedded_chunks)
            print(f"Resuming from checkpoint: {start_idx}/{total} chunks already completed.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint, starting from 0 ({e})")
            embedded_chunks = []
            start_idx = 0

    i = start_idx
    while i < total:
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        success = False
        attempts = 0
        while not success and attempts < 10:
            try:
                attempts += 1
                vectors = call_embed(texts)
                for chunk, vec in zip(batch, vectors):
                    chunk_copy = dict(chunk)
                    chunk_copy["embedding"] = np.array(vec, dtype=np.float32)
                    embedded_chunks.append(chunk_copy)

                success = True
                completed = len(embedded_chunks)
                print(f"  [Embeddings] Processed {completed}/{total} chunks ({completed/total*100:.1f}%)", flush=True)

                # Save checkpoint after every successful batch
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump(embedded_chunks, f)

                # Safe pace to stay comfortably under 100 RPM quota
                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait_time = 50
                    print(f"\n  [Quota Notice] Per-minute rate limit reached. Pausing {wait_time}s for quota reset...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"API attempt {attempts} failed: {e}. Retrying in 10s...")
                    time.sleep(10.0)

        if not success:
            raise RuntimeError(f"Failed to embed batch starting at index {i} after {attempts} attempts.")

        i += batch_size

    # Clean up checkpoint on completion
    if CHECKPOINT_FILE.exists():
        try:
            CHECKPOINT_FILE.unlink()
        except Exception:
            pass

    print(f"Successfully generated embeddings for all {len(embedded_chunks)} chunks.")
    return embedded_chunks


def embed_single_text(text: str) -> np.ndarray:
    """Embed a single query string using Gemini embedding model with retry."""
    attempts = 0
    while attempts < 5:
        try:
            vectors = call_embed([text])
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                time.sleep(45)
            else:
                time.sleep(2)
            attempts += 1
    raise RuntimeError("Failed to embed query text after retries.")


def save_embeddings_cache(chunks: List[Dict], cache_file: str = CACHE_FILE) -> Path:
    """Save chunks with embeddings to disk cache."""
    cache_path = OUTPUTS_DIR / cache_file
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {cache_path}")
    return cache_path


def load_embeddings_cache(cache_file: str = CACHE_FILE) -> Optional[List[Dict]]:
    """Load chunks with embeddings from disk cache."""
    cache_path = OUTPUTS_DIR / cache_file
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from cache ({cache_path})")
        return chunks
    return None


def embeddings_cache_exists(cache_file: str = CACHE_FILE) -> bool:
    """Check if cache file exists."""
    return (OUTPUTS_DIR / cache_file).exists()

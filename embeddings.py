"""Local Ollama embedding generation and model-specific cache management."""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    CACHE_FILE,
    CHECKPOINT_FILE,
    BATCH_EMBED_SIZE,
    OUTPUTS_DIR,
    REQUEST_DELAY_SECONDS,
    EMBEDDING_MODEL,
    call_embed,
)


def embed_chunks(
    chunks: List[Dict],
    batch_size: int = BATCH_EMBED_SIZE
) -> List[Dict]:

    total = len(chunks)
    embedded_chunks: List[Dict] = []
    start_idx = 0

    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                checkpoint = pickle.load(f)

            if checkpoint.get("embedding_model") == EMBEDDING_MODEL:
                embedded_chunks = checkpoint["chunks"]
                start_idx = len(embedded_chunks)

                print(
                    f"Resuming {EMBEDDING_MODEL} embedding checkpoint: "
                    f"{start_idx}/{total}"
                )

        except Exception as exc:
            print(
                f"Warning: ignoring unusable embedding checkpoint: {exc}"
            )

    print(
        f"Embedding {total} chunks with Ollama model "
        f"'{EMBEDDING_MODEL}' (batch={batch_size})"
    )

    i = start_idx

    while i < total:

        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            vectors = call_embed(texts)

        except Exception as batch_error:

            print(
                f"\n[Embedding batch failure] "
                f"chunks {i + 1}-{i + len(batch)}"
            )
            print(f"Reason: {batch_error}")
            print("Retrying batch one chunk at a time...")

            vectors = []

            for local_idx, text in enumerate(texts):

                global_idx = i + local_idx

                try:
                    vector = call_embed([text])[0]

                    # Explicitly reject invalid vectors.
                    vector_array = np.asarray(
                        vector,
                        dtype=np.float32
                    )

                    if not np.all(np.isfinite(vector_array)):
                        raise RuntimeError(
                            "Embedding contains NaN or infinite values"
                        )

                    vectors.append(vector)

                except Exception as chunk_error:

                    chunk = batch[local_idx]

                    print("\n" + "=" * 70)
                    print("[FAILED CHUNK]")
                    print(f"Index: {global_idx}")
                    print(f"Source: {chunk.get('source')}")
                    print(
                        f"Section: "
                        f"{chunk.get('section_hint')}"
                    )
                    print(f"Characters: {len(text)}")
                    print(f"Error: {chunk_error}")
                    print("-" * 70)
                    print("First 1000 characters:")
                    print(text[:1000])
                    print("=" * 70)

                    raise RuntimeError(
                        f"Embedding failed for chunk {global_idx} "
                        f"from {chunk.get('source')} "
                        f"section {chunk.get('section_hint')}"
                    ) from chunk_error

        # Validate the entire successful batch.
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding batch mismatch at {i}: "
                f"{len(vectors)} != {len(batch)}"
            )

        for chunk, vector in zip(batch, vectors):

            vector_array = np.asarray(
                vector,
                dtype=np.float32
            )

            if not np.all(np.isfinite(vector_array)):
                raise RuntimeError(
                    f"Invalid embedding detected for chunk "
                    f"{chunk.get('section_hint')}"
                )

            item = dict(chunk)
            item["embedding"] = vector_array
            embedded_chunks.append(item)

        # Save checkpoint after every successful batch.
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "chunks": embedded_chunks,
                },
                f
            )

        done = len(embedded_chunks)

        print(
            f"  [Embeddings] {done}/{total} "
            f"({done / max(total, 1) * 100:.1f}%)",
            flush=True
        )

        if REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS)

        i += len(batch)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    return embedded_chunks


def embed_single_text(text: str) -> np.ndarray:
    vectors = call_embed([text])
    return np.asarray(vectors[0], dtype=np.float32)


def save_embeddings_cache(chunks: List[Dict], cache_file: str = CACHE_FILE) -> Path:
    path = OUTPUTS_DIR / cache_file
    with open(path, "wb") as f:
        pickle.dump({"embedding_model": EMBEDDING_MODEL, "chunks": chunks}, f)
    print(f"Saved {len(chunks)} chunks using {EMBEDDING_MODEL} to {path}")
    return path


def load_embeddings_cache(cache_file: str = CACHE_FILE) -> Optional[List[Dict]]:
    path = OUTPUTS_DIR / cache_file
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if data.get("embedding_model") != EMBEDDING_MODEL:
            print(f"Ignoring cache built with '{data.get('embedding_model')}', expected '{EMBEDDING_MODEL}'")
            return None
        return data.get("chunks")

    # Legacy list cache: accept only if the caller explicitly keeps the same model.
    print("Ignoring legacy embedding cache without model metadata; rebuild is safer.")
    return None


def embeddings_cache_exists(cache_file: str = CACHE_FILE) -> bool:
    return (OUTPUTS_DIR / cache_file).exists()

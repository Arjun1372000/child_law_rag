"""Rebuild the local embedding index from the current law_data corpus."""

from config import CACHE_FILE, OUTPUTS_DIR, EMBEDDING_MODEL
from document_processing import load_documents, chunk_documents
from embeddings import embed_chunks, save_embeddings_cache


def main() -> None:
    cache = OUTPUTS_DIR / CACHE_FILE
    checkpoint = OUTPUTS_DIR / f"{cache.stem.replace('_cache','')}_checkpoint.pkl"

    if cache.exists():
        cache.unlink()
        print(f"Removed old cache: {cache}")
    if checkpoint.exists():
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint}")

    documents = load_documents()
    required = {
        "pocso_act_2012_official.pdf",
        "jjact2015.pdf",
    }
    present = {d["file"] for d in documents}
    missing = required - present
    if missing:
        raise FileNotFoundError(
            "Missing authoritative corpus file(s): " + ", ".join(sorted(missing)) +
            ". Add the official PDFs to law_data before rebuilding."
        )

    chunks = chunk_documents(documents)
    chunks = embed_chunks(chunks)
    save_embeddings_cache(chunks)
    print(f"\nIndex rebuilt successfully with embedding model: {EMBEDDING_MODEL}")
    print(f"Chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()

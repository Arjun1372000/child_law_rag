"""Quick end-to-end smoke test for the Ollama-backed pipeline."""
from document_processing import load_documents, chunk_documents
from embeddings import embed_chunks
from config import call_llm

# Step 1: Load and chunk
docs = load_documents()
chunks = chunk_documents(docs)
print(f"\nTotal chunks: {len(chunks)}")

# Step 2: Embed just the first 5 chunks as a quick test
test_chunks = chunks[:5]
embedded = embed_chunks(test_chunks, batch_size=5)
print(f"Embedded {len(embedded)} chunks")
dim = len(embedded[0]["embedding"])
print(f"Embedding dimension: {dim}")

# Step 3: Test LLM with system instruction
answer = call_llm(
    "What age does POCSO define as a child?",
    system_instruction="You are a legal assistant. Answer briefly."
)
print(f"\nLLM answer: {answer[:300]}")
print("\n=== ALL PIPELINE TESTS PASSED ===")


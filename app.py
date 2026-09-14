"""
Child Law Legal Assistant - Main Application
______________________________________________________________
Design and Development of an AI-Powered Legal Assistant for Child Protection and Rights Awareness.
Focus: Central Indian Child Protection Legislation (POCSO Act 2012, Juvenile Justice Act 2015, Constitution).
"""

from config import LAST_RETRIEVAL_SCORES
from document_processing import load_documents, chunk_documents
from embeddings import (
    embed_chunks,
    save_embeddings_cache,
    load_embeddings_cache,
    embeddings_cache_exists
)
from retrieval import (
    retrieve_bm25,
    retrieve_dense,
    retrieve_hybrid_rrf,
    retrieve_reranked,
    retrieve_crag_hyde
)
from answer_generation import (
    validate_child_law_domain,
    generate_answer
)
from evaluation import run_comprehensive_benchmark


def main():
    print("=" * 70)
    print("  AI-Powered Legal Assistant for Child Protection & Rights Awareness")
    print("  Focus: Central Indian Legislation (POCSO 2012, JJ Act 2015, Constitution)")
    print("=" * 70)

    # Load embeddings from cache or generate
    if embeddings_cache_exists():
        print("Loading statutory chunks from disk cache...")
        chunks = load_embeddings_cache()
    else:
        print("First-time setup: Ingesting central legal statutes...")
        documents = load_documents()
        chunks = chunk_documents(documents)
        print("Generating embeddings via Google Gemini...")
        chunks = embed_chunks(chunks)
        save_embeddings_cache(chunks)

    print(f"✓ Ready: Corpus contains {len(chunks)} structured statutory chunks.\n")

    conversation_history = []
    current_method = None

    methods_map = {
        "1": ("BM25 (Lexical Match)", retrieve_bm25),
        "2": ("Dense (Gemini Embeddings)", retrieve_dense),
        "3": ("Hybrid (Reciprocal Rank Fusion - RRF)", retrieve_hybrid_rrf),
        "4": ("Hybrid + Cross-Encoder Reranking", retrieve_reranked),
        "5": ("Corrective RAG (HyDE)", retrieve_crag_hyde),
    }

    while True:
        if not current_method:
            print("\nSelect Retrieval Strategy:")
            print("  1. Lexical Search (BM25 Okapi)")
            print("  2. Dense Vector Retrieval (Gemini Embeddings)")
            print("  3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)")
            print("  4. Two-Stage Reranked Hybrid (Cross-Scoring)")
            print("  5. Corrective RAG with HyDE (Hypothetical Document Embeddings)")
            print("  6. Run Scientific Benchmark & Generate Paper Tables")
            print("  7. Exit")

            choice = input("\nEnter choice (1-7): ").strip()

            if choice == "6":
                run_comprehensive_benchmark(chunks)
                continue

            if choice == "7":
                print("Exiting assistant. Goodbye!")
                break

            if choice not in methods_map:
                print("Invalid choice. Please select 1 through 7.")
                continue

            current_method = choice
            method_label, _ = methods_map[current_method]
            print(f"\n[Active Strategy: {method_label}]")

        question = input("\nEnter your legal question (or 'back', 'clear', 'exit'): ").strip()

        if not question:
            continue

        if question.lower() == "back":
            current_method = None
            continue

        if question.lower() == "clear":
            conversation_history = []
            print("Conversation memory cleared.")
            continue

        if question.lower() == "exit":
            print("Exiting assistant. Goodbye!")
            break

        # Dual-Stage Domain Guardrail Check
        is_valid, category, redirection = validate_child_law_domain(question)
        if not is_valid:
            print("\n" + redirection + "\n")
            continue

        # Retrieve relevant statutory context
        method_label, method_fn = methods_map[current_method]
        print(f"\nRetrieving via {method_label}...")
        retrieved = method_fn(question, chunks)

        print("\n" + "-" * 70)
        print("Retrieved Statutory Provisions:")
        for idx, chunk in enumerate(retrieved):
            key = (chunk["file"], chunk["chunk_id"])
            score = LAST_RETRIEVAL_SCORES.get(key, 0.0)
            act_title = chunk.get("act_title", chunk["file"])
            section_hint = chunk.get("section_hint", "General")
            print(f"  [{idx+1}] {act_title} | {section_hint} (Score: {score:.3f})")
        print("-" * 70)

        # Generate Grounded Answer
        print("\nGenerating grounded legal response...\n")
        answer = generate_answer(question, retrieved, conversation_history)
        print(answer)
        print("\n" + "=" * 70)

        # Context Memory (last 3 turns)
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]


if __name__ == "__main__":
    main()

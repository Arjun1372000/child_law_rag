"""
Child Law Legal Assistant - Main Application
______________________________________________________________

A specialized legal assistant focused on child protection and child rights laws.
Supports multiple retrieval strategies including dense, hybrid, multi-query, and corrective retrieval.

Module Organization:
  - config.py: Configuration, constants, and global state
  - document_processing.py: Loading and chunking documents
  - embeddings.py: Embedding creation and caching
  - retrieval.py: Retrieval methods (dense, hybrid, multi-query, corrective)
  - answer_generation.py: Answer generation and filtering
  - evaluation.py: Evaluation and visualization functions
"""

from pathlib import Path

# imports from local modules
from config import LAST_RETRIEVAL_SCORES
from document_processing import load_documents, chunk_documents
from embeddings import (
    embed_chunks,
    save_embeddings_cache,
    load_embeddings_cache,
    embeddings_cache_exists
)
from retrieval import (
    retrieve_dense,
    retrieve_hybrid,
    retrieve_multiquery,
    retrieve_corrective
)
from answer_generation import (
    generate_answer,
    is_child_law_question,
    filter_chunks_for_state
)
from evaluation import run_evaluation




def main():    
    # Load embeddings from cache or create new ones
    if embeddings_cache_exists():
        print("Loading chunks from cache...")
        chunks = load_embeddings_cache()
    else:
        print("Loading documents...")
        documents = load_documents()

        print("Chunking documents...")
        chunks = chunk_documents(documents)

        print("Creating embeddings...")
        chunks = embed_chunks(chunks)
        
        print("Saving embeddings cache...")
        save_embeddings_cache(chunks)

    print("\n" + "="*60)
    print("Child Law Legal Assistant - Conversation Mode")
    print("="*60)
    print("Type 'exit' to quit | 'clear' to reset conversation history")
    print("="*60 + "\n")

    # Conversation history for context window
    conversation_history = []
    current_method = None

    while True:
        if not current_method:
            print("\nChoose retrieval method:")
            print("1. Dense")
            print("2. Hybrid")
            print("3. Multi Query")
            print("4. Corrective")
            print("5. Run Evaluation")
            print("6. Exit")

            choice = input("\nEnter choice: ").strip()

            if choice == "5":
                run_evaluation(chunks)
                continue

            if choice == "6":
                print("Goodbye!")
                break

            if choice not in ["1", "2", "3", "4"]:
                print("Invalid choice. Please try again.")
                continue

            current_method = choice

        question = input("\nYour question (or 'back'/'clear'/'exit'): ").strip()

        if question.lower() == "back":
            current_method = None
            continue

        if question.lower() == "clear":
            conversation_history = []
            print("\nConversation history cleared!")
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not is_child_law_question(question):
            print("\nThis assistant is specialized only for child protection and child rights related legal questions.")
            continue

        filtered_chunks = filter_chunks_for_state(question, chunks)

        if current_method == "1":
            retrieved = retrieve_dense(question, filtered_chunks)
        elif current_method == "2":
            retrieved = retrieve_hybrid(question, filtered_chunks)
        elif current_method == "3":
            retrieved = retrieve_multiquery(question, filtered_chunks)
        elif current_method == "4":
            retrieved = retrieve_corrective(question, filtered_chunks)

        print("\n" + "-"*60)
        print("Retrieved Sources (with Retrieval Scores):")
        for chunk in retrieved:
            chunk_key = (chunk['file'], chunk['chunk_id'])
            score = LAST_RETRIEVAL_SCORES.get(chunk_key, 0.0)
            # Format score as percentage or decimal
            if score > 1.0:
                score_display = f"{score:.3f}"
            else:
                score_display = f"{score:.1%}"
            print(f"  • {chunk['file']} (chunk {chunk['chunk_id']}) [Score: {score_display}]")
        print("-"*60)

        print("\nAnswer:\n")
        answer = generate_answer(question, retrieved, conversation_history)
        print(answer)

        # Add to conversation history for context window
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        # Keep only last 6 messages (3 q&a pairs) for context window
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]


if __name__ == "__main__":
    main()


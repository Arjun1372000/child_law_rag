# answer_generation.py
"""
Functions for generating answers and filtering based on domain and location
"""

from typing import List, Dict, Optional
from config import client, LLM_MODEL, CITATION_MAP, CHILD_LAW_KEYWORDS, STATE_KEYWORDS


def is_child_law_question(question: str) -> bool:
    """
    Check if the question is related to child law
    
    Args:
        question: The question to check
        
    Returns:
        True if question is about child law, False otherwise
    """
    q = question.lower()
    return any(k in q for k in CHILD_LAW_KEYWORDS)


def filter_chunks_for_state(question: str, chunks: List[Dict]) -> List[Dict]:
    """
    Filter chunks based on state-specific keywords in the question
    
    Args:
        question: The question to analyze
        chunks: List of all chunks
        
    Returns:
        Filtered list of chunks relevant to the state mentioned
    """
    q = question.lower()

    for state, keywords in STATE_KEYWORDS.items():
        if any(k in q for k in keywords):
            state_chunks = [c for c in chunks if state in c["file"].lower()]
            return state_chunks if state_chunks else chunks

    return chunks


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate an answer to a question using retrieved chunks and conversation history
    
    Args:
        question: The question to answer
        retrieved_chunks: List of retrieved relevant chunks
        conversation_history: Optional conversation history for context
        
    Returns:
        Generated answer text
    """
    CITATION_MAP.clear()
    
    if not retrieved_chunks:
        return "Insufficient legal information found in the available documents."

    # Build context with citation references
    context_lines = []
    for idx, chunk in enumerate(retrieved_chunks):
        citation_ref = f"[{chunk['file']} chunk {chunk['chunk_id']}]"
        CITATION_MAP[f"CITE{idx}"] = citation_ref
        context_lines.append(f"[CITE{idx}] {chunk['text'][:100]}...\n{chunk['text']}")
    
    context = "\n\n".join(context_lines)

    # Build conversation context window
    messages = []
    
    # Add previous conversation history if available
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current question
    prompt = f"""
You are a legal assistant specialized only in child protection and child rights laws.

Answer ONLY from the legal context below.
Do not invent section numbers or laws.
When citing sources, use the format [FileName chunk N].
If the answer is not available, reply exactly:
Insufficient legal information found in the available documents.
The answer needs to be formatted in a way that it can be easily read by a layperson, but also include the relevant legal citations.
The answer needs to be formatted for terminal display, with clear paragraphs and line breaks.

Legal Context:
{context}

Question:
{question}

Provide citations like [TamilNaduRules.pdf chunk 3] right after relevant statements.
"""
    
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content

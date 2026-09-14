# retrieval.py
"""
Comparative Information Retrieval Suite for Indian Child Law RAG:
1. Lexical Baseline: BM25 (Okapi)
2. Dense Baseline: Semantic Vector Cosine Similarity
3. Hybrid with Reciprocal Rank Fusion (RRF)
4. Two-Stage Reranked Hybrid (Cross-Scoring)
5. Corrective RAG with Hypothetical Document Embeddings (HyDE)
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

from config import (
    call_llm,
    LAST_RETRIEVAL_SCORES,
    DEFAULT_TOP_K,
    RRF_K,
    BM25_K1,
    BM25_B
)
from embeddings import embed_single_text

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize_legal_text(text: str) -> List[str]:
    """Tokenize legal text with statutory term preservation (e.g., 'section', numbers, keywords)."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in clean.split() if len(t) > 1]


# -------------------------------------------------------------------------
# Method 1: Lexical BM25 Retrieval
# -------------------------------------------------------------------------
def retrieve_bm25(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Lexical BM25 retrieval for exact statutory terminology and section numbers.
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    tokenized_corpus = [tokenize_legal_text(c["text"]) for c in chunks]
    query_tokens = tokenize_legal_text(question)
    
    if not query_tokens:
        return chunks[:top_k]

    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    scores = bm25.get_scores(query_tokens)

    # Normalize scores to 0.0 - 1.0
    max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
    scored = []
    for idx, chunk in enumerate(chunks):
        norm_score = float(scores[idx] / max_score)
        chunk_key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[chunk_key] = norm_score
        scored.append((norm_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# -------------------------------------------------------------------------
# Method 2: Dense Semantic Vector Retrieval
# -------------------------------------------------------------------------
def retrieve_dense(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Dense semantic retrieval using Gemini embeddings and cosine similarity.
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    q_embedding = embed_single_text(question)
    # Normalize query vector for cosine similarity
    q_norm = np.linalg.norm(q_embedding)
    if q_norm > 0:
        q_vec = q_embedding / q_norm
    else:
        q_vec = q_embedding

    scored = []
    for chunk in chunks:
        doc_vec = chunk["embedding"]
        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm > 0:
            doc_norm_vec = doc_vec / doc_norm
            similarity = float(np.dot(q_vec, doc_norm_vec))
        else:
            similarity = 0.0

        # Rescale cosine (-1 to 1) to (0 to 1)
        rescaled_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        chunk_key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[chunk_key] = rescaled_score
        scored.append((rescaled_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# -------------------------------------------------------------------------
# Method 3: Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
# -------------------------------------------------------------------------
def retrieve_hybrid_rrf(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Hybrid retrieval fusing BM25 lexical rank and Dense vector rank via Reciprocal Rank Fusion (RRF).
    Standard formulation: RRF(d) = sum(1 / (k + rank_i(d))) where k = 60.
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    pool_size = min(len(chunks), max(top_k * 4, 25))
    
    # Run first-stage dense and lexical ranking
    dense_candidates = retrieve_dense(question, chunks, top_k=pool_size)
    dense_scores = dict(LAST_RETRIEVAL_SCORES)
    
    bm25_candidates = retrieve_bm25(question, chunks, top_k=pool_size)
    bm25_scores = dict(LAST_RETRIEVAL_SCORES)

    rrf_scores = defaultdict(float)

    # Accumulate RRF for dense ranks
    for rank, chunk in enumerate(dense_candidates):
        key = (chunk["file"], chunk["chunk_id"])
        rrf_scores[key] += 1.0 / (RRF_K + (rank + 1))

    # Accumulate RRF for BM25 ranks
    for rank, chunk in enumerate(bm25_candidates):
        key = (chunk["file"], chunk["chunk_id"])
        rrf_scores[key] += 1.0 / (RRF_K + (rank + 1))

    # Index chunks by key
    chunk_map = {(c["file"], c["chunk_id"]): c for c in chunks}
    
    scored = []
    for key, score in rrf_scores.items():
        chunk = chunk_map.get(key)
        if chunk:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Store top scores for UI/inspection
    for score, chunk in scored[:top_k]:
        key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[key] = score

    return [chunk for _, chunk in scored[:top_k]]


# -------------------------------------------------------------------------
# Method 4: Two-Stage Hybrid + Cross-Encoder Reranking
# -------------------------------------------------------------------------
def retrieve_reranked(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Two-Stage Retrieval:
    Stage 1: Retrieve candidate pool using Hybrid RRF.
    Stage 2: Pointwise LLM cross-encoder scoring to evaluate exact statutory relevance.
    """
    LAST_RETRIEVAL_SCORES.clear()
    candidate_k = min(len(chunks), top_k * 3)
    candidates = retrieve_hybrid_rrf(question, chunks, top_k=candidate_k)

    if len(candidates) <= top_k:
        return candidates

    scored = []
    for chunk in candidates:
        text_preview = chunk["text"][:600]
        act = chunk.get("act_title", chunk["file"])
        hint = chunk.get("section_hint", "General")
        
        # Cross-encoder prompt
        scoring_prompt = f"""Rate how directly relevant this legal statutory passage is to answering the user question on Indian child law.
Score from 0 (completely irrelevant) to 10 (directly answers the question).
Respond with ONLY an integer from 0 to 10.

Question: {question}
Statute: {act} ({hint})
Passage:
{text_preview}"""

        try:
            res = call_llm(scoring_prompt).strip()
            digits = re.findall(r"\b(10|[0-9])\b", res)
            score = float(digits[0]) if digits else 5.0
        except Exception:
            score = 5.0

        key = (chunk["file"], chunk["chunk_id"])
        norm_score = score / 10.0
        LAST_RETRIEVAL_SCORES[key] = norm_score
        scored.append((norm_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# -------------------------------------------------------------------------
# Method 5: Corrective RAG with Hypothetical Document Embeddings (HyDE)
# -------------------------------------------------------------------------
def generate_hypothetical_statute(question: str) -> str:
    """Draft a hypothetical Indian statutory clause to bridge vocabulary mismatch."""
    prompt = f"""Generate a concise hypothetical Indian statutory excerpt (in formal legal drafting style with section language) that would answer this legal query:
Query: {question}

Respond with only 2-3 sentences of hypothetical statutory text:"""
    try:
        return call_llm(prompt).strip()
    except Exception:
        return question


def retrieve_crag_hyde(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Corrective RAG using HyDE:
    1. Generates a hypothetical statutory clause.
    2. Embeds both question and hypothetical legal passage.
    3. Retrieves from hybrid fusion.
    4. Applies confidence grading to ensure retrieved context relevance.
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    hypothetical_doc = generate_hypothetical_statute(question)
    combined_query = f"{question}\n{hypothetical_doc}"

    # Retrieve using RRF on the enhanced query
    retrieved = retrieve_hybrid_rrf(combined_query, chunks, top_k=top_k)

    # Quality check: If no high scoring chunks, fall back to exact lexical query
    max_score = max(LAST_RETRIEVAL_SCORES.values()) if LAST_RETRIEVAL_SCORES else 0.0
    if max_score < 0.015:  # Low RRF threshold indicating sparse match
        lexical_fallback = retrieve_bm25(question, chunks, top_k=top_k)
        return lexical_fallback

    return retrieved


# Backward compatibility aliases
retrieve_hybrid = retrieve_hybrid_rrf
retrieve_corrective = retrieve_crag_hyde
retrieve_multiquery = retrieve_crag_hyde

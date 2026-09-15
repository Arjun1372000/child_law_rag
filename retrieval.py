"""Retrieval strategies for Indian Child Law RAG."""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from config import (
    LAST_RETRIEVAL_SCORES,
    DEFAULT_TOP_K,
    RRF_K,
    BM25_K1,
    BM25_B,
)

from embeddings import embed_single_text

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def tokenize_legal_text(text: str) -> List[str]:
    text = re.sub(r"[^\w\s()/-]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


def _chunk_key(chunk: Dict) -> Tuple[str, int]:
    return chunk["file"], chunk["chunk_id"]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def retrieve_bm25(
    question: str,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:

    LAST_RETRIEVAL_SCORES.clear()

    if not question or not chunks:
        return chunks[:top_k]

    if BM25Okapi is None:
        raise ImportError("rank-bm25 is required for BM25 retrieval")

    tokenized_chunks = [
        tokenize_legal_text(c.get("text", ""))
        for c in chunks
    ]

    query_tokens = tokenize_legal_text(question)

    if not query_tokens:
        return chunks[:top_k]

    bm25 = BM25Okapi(
        tokenized_chunks,
        k1=BM25_K1,
        b=BM25_B,
    )

    scores = bm25.get_scores(query_tokens)

    max_score = float(np.max(scores)) if len(scores) else 0.0
    denom = max_score if max_score > 0 else 1.0

    ranked = []

    for idx, chunk in enumerate(chunks):
        normalized_score = max(
            0.0,
            float(scores[idx]) / denom,
        )

        LAST_RETRIEVAL_SCORES[_chunk_key(chunk)] = normalized_score
        ranked.append((normalized_score, chunk))

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [chunk for _, chunk in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def retrieve_dense(
    question: str,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:

    LAST_RETRIEVAL_SCORES.clear()

    if not question or not chunks:
        return chunks[:top_k]

    query_vector = embed_single_text(question)
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return chunks[:top_k]

    query_vector = query_vector / query_norm

    ranked = []

    for chunk in chunks:
        vector = np.asarray(
            chunk["embedding"],
            dtype=np.float32,
        )

        vector_norm = np.linalg.norm(vector)

        if vector_norm == 0:
            score = 0.0
        else:
            cosine = float(
                np.dot(
                    query_vector,
                    vector / vector_norm,
                )
            )
            score = (cosine + 1.0) / 2.0

        LAST_RETRIEVAL_SCORES[_chunk_key(chunk)] = score
        ranked.append((score, chunk))

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [chunk for _, chunk in ranked[:top_k]]


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: List[List[Dict]],
    top_k: int,
) -> List[Dict]:

    rrf_scores = defaultdict(float)
    by_key = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = _chunk_key(chunk)

            by_key[key] = chunk
            rrf_scores[key] += 1.0 / (RRF_K + rank)

    ranked = sorted(
        (
            (score, by_key[key])
            for key, score in rrf_scores.items()
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    return [chunk for _, chunk in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid RRF
# ---------------------------------------------------------------------------

def retrieve_hybrid_rrf(
    question: str,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:

    LAST_RETRIEVAL_SCORES.clear()

    if not chunks:
        return []

    candidate_k = min(
        len(chunks),
        max(25, top_k * 4),
    )

    dense_results = retrieve_dense(
        question,
        chunks,
        candidate_k,
    )

    bm25_results = retrieve_bm25(
        question,
        chunks,
        candidate_k,
    )

    results = _rrf_fuse(
        [dense_results, bm25_results],
        top_k,
    )

    # Reconstruct RRF scores for the returned documents.
    score_map = defaultdict(float)

    for ranked_list in [dense_results, bm25_results]:
        for rank, chunk in enumerate(ranked_list, start=1):
            score_map[_chunk_key(chunk)] += 1.0 / (
                RRF_K + rank
            )

    for chunk in results:
        LAST_RETRIEVAL_SCORES[_chunk_key(chunk)] = score_map[
            _chunk_key(chunk)
        ]

    return results


# ---------------------------------------------------------------------------
# Local reranking
# ---------------------------------------------------------------------------

def _exact_legal_bonus(
    question: str,
    chunk: Dict,
) -> float:

    q = question.lower()
    text = chunk.get("text", "").lower()

    bonus = 0.0

    # Explicit provision references are extremely important
    # in statutory retrieval.
    provisions = re.findall(
        r"\b(?:section|article|rule)\s+\d+[a-z]?(?:\(\d+\))?",
        q,
    )

    for provision in provisions:
        normalized = provision.replace(" ", "")
        normalized_text = text.replace(" ", "")

        if normalized in normalized_text:
            bonus += 0.20

    # Exact act-title clues.
    act_title = chunk.get("act_title", "").lower()

    for term in [
        "pocso",
        "juvenile justice",
        "constitution",
        "child",
    ]:
        if term in q and term in act_title:
            bonus += 0.05

    return min(bonus, 0.30)


def retrieve_reranked(
    question: str,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:

    LAST_RETRIEVAL_SCORES.clear()

    candidate_k = min(
        len(chunks),
        max(30, top_k * 6),
    )

    candidates = retrieve_hybrid_rrf(
        question,
        chunks,
        candidate_k,
    )

    if not candidates:
        return []

    query_tokens = set(
        tokenize_legal_text(question)
    )

    query_vector = embed_single_text(question)
    query_norm = np.linalg.norm(query_vector)

    if query_norm:
        query_vector = query_vector / query_norm

    reranked = []

    for original_rank, chunk in enumerate(candidates, start=1):

        text_tokens = set(
            tokenize_legal_text(
                chunk.get("text", "")
            )
        )

        lexical_overlap = (
            len(query_tokens & text_tokens)
            / max(1, len(query_tokens))
        )

        vector = np.asarray(
            chunk["embedding"],
            dtype=np.float32,
        )

        vector_norm = np.linalg.norm(vector)

        if vector_norm and query_norm:
            cosine = float(
                np.dot(
                    query_vector,
                    vector / vector_norm,
                )
            )
            dense_score = (cosine + 1.0) / 2.0
        else:
            dense_score = 0.0

        # Mild preference for the original hybrid rank.
        rank_score = 1.0 / np.sqrt(original_rank)

        exact_bonus = _exact_legal_bonus(
            question,
            chunk,
        )

        score = (
            0.50 * dense_score
            + 0.20 * lexical_overlap
            + 0.20 * rank_score
            + exact_bonus
        )

        LAST_RETRIEVAL_SCORES[_chunk_key(chunk)] = score
        reranked.append((score, chunk))

    reranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        chunk
        for _, chunk in reranked[:top_k]
    ]


# ---------------------------------------------------------------------------
# Corrective retrieval
# ---------------------------------------------------------------------------

LEGAL_QUERY_EXPANSIONS = {
    "punishment": ["penalty", "sentence", "imprisonment"],
    "sentence": ["punishment", "penalty", "imprisonment"],
    "report": ["reporting", "mandatory reporting", "police"],
    "police": ["police officer", "special juvenile police unit"],
    "bail": ["release", "bail order"],
    "foster care": ["foster", "placement", "care"],
    "abandoned": ["orphaned", "surrendered", "child in need of care and protection"],
    "education": ["free education", "compulsory education", "school"],
    "child labour": ["employment of children", "hazardous employment"],
    "corporal punishment": ["physical punishment", "child care institution"],
    "sexual harassment": ["sexual intent", "harassment of child"],
    "pornographic": ["pornography", "pornographic material"],
}


def _expand_legal_query(question: str) -> str:

    lower = question.lower()
    additions = []

    for trigger, expansions in LEGAL_QUERY_EXPANSIONS.items():
        if trigger in lower:
            additions.extend(expansions)

    if not additions:
        return question

    unique_additions = list(dict.fromkeys(additions))

    return (
        question
        + " "
        + " ".join(unique_additions)
    )


def _retrieval_confidence(
    retrieved: List[Dict],
    question: str,
) -> float:

    if not retrieved:
        return 0.0

    scores = []

    for chunk in retrieved[:3]:

        text = chunk.get("text", "").lower()
        query_terms = tokenize_legal_text(question)

        if not query_terms:
            continue

        text_terms = set(
            tokenize_legal_text(text)
        )

        overlap = len(
            set(query_terms) & text_terms
        ) / max(1, len(set(query_terms)))

        scores.append(overlap)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def retrieve_crag_hyde(
    question: str,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict]:
    """
    Corrective retrieval without an additional LLM call.

    Stage 1:
        Hybrid BM25 + dense retrieval.

    Stage 2:
        Estimate retrieval confidence.

    Stage 3:
        If confidence is weak, reformulate the query using
        domain-specific legal terminology and perform a second
        hybrid retrieval.

    Stage 4:
        Fuse the original and corrective retrieval results.

    This keeps the method fully local and reproducible while
    making it genuinely different from ordinary Hybrid RRF.
    """

    LAST_RETRIEVAL_SCORES.clear()

    initial = retrieve_hybrid_rrf(
        question,
        chunks,
        top_k=top_k,
    )

    if not initial:
        return []

    confidence = _retrieval_confidence(
        initial,
        question,
    )

    # High-confidence retrieval needs no correction.
    if confidence >= 0.35:
        return initial

    corrected_query = _expand_legal_query(
        question
    )

    # Expansion did not change the query.
    if corrected_query == question:
        return initial

    candidate_k = min(
        len(chunks),
        max(25, top_k * 4),
    )

    original_dense = retrieve_dense(
        question,
        chunks,
        candidate_k,
    )

    original_bm25 = retrieve_bm25(
        question,
        chunks,
        candidate_k,
    )

    corrected_dense = retrieve_dense(
        corrected_query,
        chunks,
        candidate_k,
    )

    corrected_bm25 = retrieve_bm25(
        corrected_query,
        chunks,
        candidate_k,
    )

    fused = _rrf_fuse(
        [
            original_dense,
            original_bm25,
            corrected_dense,
            corrected_bm25,
        ],
        top_k,
    )

    # Record fused scores.
    score_map = defaultdict(float)

    for ranked_list in [
        original_dense,
        original_bm25,
        corrected_dense,
        corrected_bm25,
    ]:
        for rank, chunk in enumerate(
            ranked_list,
            start=1,
        ):
            score_map[_chunk_key(chunk)] += 1.0 / (
                RRF_K + rank
            )

    for chunk in fused:
        LAST_RETRIEVAL_SCORES[_chunk_key(chunk)] = score_map[
            _chunk_key(chunk)
        ]

    return fused


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

retrieve_hybrid = retrieve_hybrid_rrf
retrieve_corrective = retrieve_crag_hyde
retrieve_multiquery = retrieve_crag_hyde
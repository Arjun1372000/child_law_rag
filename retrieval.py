# retrieval.py
"""
Functions for different retrieval strategies
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from config import (
    client, 
    LAST_RETRIEVAL_SCORES, 
    DEFAULT_TOP_K,
    LLM_MODEL,
    EMBEDDING_MODEL,
    BM25_K1,
    BM25_B,
    DENSE_WEIGHT,
    BM25_WEIGHT
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Warning: rank_bm25 not installed. Using fallback BM25 implementation.")
    BM25Okapi = None


# BM25 Implementation
class SimpleBM25:
    """Fallback BM25 implementation if rank_bm25 is not available"""
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_lengths = []
        self.idf = {}
        self.k1 = BM25_K1
        self.b = BM25_B
        
        self._precompute(corpus)
    
    def _precompute(self, corpus):
        """Precompute IDF scores"""
        all_tokens = set()
        
        for doc in corpus:
            tokens = set(doc.lower().split())
            self.doc_lengths.append(len(doc.split()))
            all_tokens.update(tokens)
        
        self.avg_length = np.mean(self.doc_lengths) if self.doc_lengths else 1
        
        for token in all_tokens:
            doc_count = sum(1 for doc in corpus if token in doc.lower())
            self.idf[token] = math.log((len(corpus) - doc_count + 0.5) / (doc_count + 0.5) + 1)
    
    def get_scores(self, query):
        """Get BM25 scores for query against all documents"""
        tokens = query.lower().split()
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            doc_tokens = doc.lower().split()
            doc_score = 0
            
            for token in tokens:
                if token in doc_tokens:
                    token_freq = doc_tokens.count(token)
                    idf_score = self.idf.get(token, 0)
                    norm_length = self.doc_lengths[idx] / self.avg_length
                    bm25_component = idf_score * (token_freq * (self.k1 + 1)) / \
                                   (token_freq + self.k1 * (1 - self.b + self.b * norm_length))
                    doc_score += bm25_component
            
            scores.append(doc_score)
        
        return scores


def get_bm25_scores(query: str, texts: List[str]) -> List[float]:
    """
    Get BM25 scores for a query against a list of texts
    
    Args:
        query: Search query
        texts: List of texts to score
        
    Returns:
        List of BM25 scores
    """
    if BM25Okapi is not None:
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        return bm25.get_scores(query.lower().split())
    else:
        bm25 = SimpleBM25(texts)
        return bm25.get_scores(query)


# Dense Retrieval
def retrieve_dense(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve chunks using dense embeddings (semantic similarity)
    
    Args:
        question: Query question
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of most relevant chunks
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )

    q_embedding = np.array(response.data[0].embedding)

    scored = []

    for chunk in chunks:
        score = float(np.dot(q_embedding, chunk["embedding"]))
        
        # Track score
        chunk_key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[chunk_key] = score
        
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored[:top_k]]


# Hybrid Retrieval (Dense + BM25)
def retrieve_hybrid(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve chunks using hybrid method (dense + BM25)
    
    Args:
        question: Query question
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of most relevant chunks
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )

    q_embedding = np.array(response.data[0].embedding)
    
    # Get BM25 scores
    texts = [chunk["text"] for chunk in chunks]
    bm25_scores = get_bm25_scores(question, texts)
    
    # Normalize both scores to 0-1 range
    dense_scores = [float(np.dot(q_embedding, chunk["embedding"])) for chunk in chunks]
    
    if len(dense_scores) > 0:
        min_dense = min(dense_scores)
        max_dense = max(dense_scores)
        dense_range = max_dense - min_dense if max_dense > min_dense else 1
        dense_scores = [(s - min_dense) / dense_range for s in dense_scores]
    
    if len(bm25_scores) > 0:
        min_bm25 = min(bm25_scores)
        max_bm25 = max(bm25_scores)
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1
        bm25_scores = [(s - min_bm25) / bm25_range for s in bm25_scores]
    
    scored = []
    for i, chunk in enumerate(chunks):
        # Weighted combination: 70% dense, 30% BM25
        final_score = DENSE_WEIGHT * dense_scores[i] + BM25_WEIGHT * bm25_scores[i]
        
        # Track score
        chunk_key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[chunk_key] = final_score
        
        scored.append((final_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored[:top_k]]


# Query Variant Generation
def generate_query_variants(question: str) -> List[str]:
    """
    Generate alternative search queries for multi-query retrieval
    
    Args:
        question: Original query
        
    Returns:
        List of query variants including the original
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Generate 3 short alternative legal search queries for: {question}"
            }
        ]
    )

    raw = response.choices[0].message.content.strip().split("\n")

    variants = [question]
    for line in raw:
        cleaned = re.sub(r"^[0-9\-\.\)\s]+", "", line).strip()
        if cleaned:
            variants.append(cleaned)

    return variants[:4]


# Multi-Query Retrieval
def retrieve_multiquery(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve chunks using multi-query strategy (multiple reformulations)
    
    Args:
        question: Query question
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of most relevant chunks
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    variants = generate_query_variants(question)

    all_results = []
    all_scores = defaultdict(float)

    for q in variants:
        results = retrieve_dense(q, chunks, top_k=top_k)
        all_results.extend(results)
        
        # Accumulate scores for each chunk
        for chunk_key, score in LAST_RETRIEVAL_SCORES.items():
            all_scores[chunk_key] += score

    unique = []
    seen = set()

    for chunk in all_results:
        key = (chunk["file"], chunk["chunk_id"])
        if key not in seen:
            seen.add(key)
            unique.append((all_scores[key], chunk))

    # Sort by accumulated score
    unique.sort(key=lambda x: x[0], reverse=True)
    
    # Update global scores for display
    LAST_RETRIEVAL_SCORES.clear()
    for score, chunk in unique[:top_k]:
        chunk_key = (chunk["file"], chunk["chunk_id"])
        LAST_RETRIEVAL_SCORES[chunk_key] = score / len(variants)  # Average score
    
    return [chunk for _, chunk in unique[:top_k]]


# Corrective Retrieval
def retrieve_corrective(question: str, chunks: List[Dict], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve chunks with fallback strategy if results are low quality
    
    Args:
        question: Query question
        chunks: List of chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of most relevant chunks
    """
    LAST_RETRIEVAL_SCORES.clear()
    
    results = retrieve_dense(question, chunks, top_k)

    if len(results) == 0:
        return retrieve_multiquery(question, chunks, top_k)

    low_relevance = all(len(chunk["text"].strip()) < 50 for chunk in results)

    if low_relevance:
        return retrieve_multiquery(question, chunks, top_k)

    return results

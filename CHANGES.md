# Child Law RAG local/evaluation fixes

## Required corpus correction
The repository's `law_data/POCSOact_pt1.pdf` is an NCPCR guidance/handbook, not the bare POCSO Act. The benchmark targets actual POCSO statutory sections, so place the official POCSO Act PDF at:

`law_data/pocso_act_2012_official.pdf`

The code intentionally refuses to silently treat the handbook as the authoritative Act.

## Ollama
Configured defaults:
- LLM: `llama3.2`
- Embeddings: `bge-m3`
- Ollama: `http://localhost:11434`

## Important
Changing from the existing `nomic-embed-text` cache to `bge-m3` requires a full index rebuild:

`python rebuild_index.py`

This creates `outputs/bge-m3_embeddings_cache.pkl` and does not reuse the previous embedding cache.

## Evaluation fixes
- Provision matching now handles `Section 2(13)`, `Section 2(14)`, `Article 21A`, etc.
- Retrieval evaluation always measures Hit@1/3/5 from five retrieved candidates.
- Target Act is checked as well as section, reducing false positives.
- Generation metrics no longer fabricate 0.85/0.90 fallback values or convert judge failures into zero.
- Default generation evaluation uses deterministic context-support and gold-answer TF-IDF similarity, so it does not need an LLM judge.
- LLM judging is optional and only used when explicitly enabled.
- LLM reranking was replaced by deterministic local reranking to eliminate dozens of Ollama calls per benchmark query.
- Ollama retries are bounded and server errors expose the HTTP response instead of hiding them.

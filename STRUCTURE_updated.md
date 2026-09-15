# Project Structure and Architecture

## 1. Overview

`child_law_rag` is a modular local Retrieval-Augmented Generation (RAG) system for Indian child-law question answering. It separates document processing, embedding generation, retrieval, answer generation, and evaluation so each stage can be developed and tested independently.

The repository has two purposes:

1. **Application:** a locally running child-law legal assistant using Ollama.
2. **Research:** a controlled comparison of retrieval strategies on a child-law benchmark.

The final research comparison uses four primary methods: BM25, Dense BGE-M3, Hybrid RRF, and Hybrid + Local Rerank.

---

## 2. High-Level Data Flow

```text
Legal Source Documents
        |
        v
document_processing.py
        |
        v
Structured Legal Chunks
        |
        v
embeddings.py
        |
        v
BGE-M3 Embeddings
        |
        v
Local Embedding Cache
        |
        v
retrieval.py
        |
+-------+-------+----------------+
|       |       |                |
BM25  Dense  Hybrid RRF   Local Rerank
|       |       |                |
+-------+-------+----------------+
        |
        v
Retrieved Top-k Legal Context
        |
        v
answer_generation.py
        |
        v
Llama 3.2 via Ollama
        |
        v
Grounded Answer + Sources

Research / Evaluation Path
        |
        v
evaluation.py
        |
        v
Benchmark + IR Metrics
        |
        v
evaluation_summary.json
        |
        v
generate_paper_figures.py
        |
        v
Publication Figures + Main Table
```

---

## 3. Repository Layout

```text
child_law_rag/
├── app.py                         # Interactive local legal assistant
├── config.py                      # Paths, models, Ollama settings, hyperparameters
├── document_processing.py         # PDF loading and statutory chunking
├── embeddings.py                  # Local embedding generation and cache management
├── retrieval.py                   # BM25, Dense, Hybrid RRF, reranking and corrective retrieval
├── answer_generation.py           # Domain validation and grounded answer generation
├── evaluation.py                  # Benchmark execution and retrieval metrics
├── generate_paper_figures.py      # Paper figures and main comparison table
├── test_ollama.py                 # Ollama connectivity/model smoke test
├── requirements.txt               # Python dependencies
├── .env.example                   # Example configuration
├── README.md                      # Project overview and usage
├── STRUCTURE.md                   # This architecture document
├── RESEARCH_PAPER.md              # Research-paper manuscript
├── diagram.md                     # Architecture diagram description
├── diagramv2.md                   # Alternate architecture diagram
│
├── law_data/                      # Legal corpus
│   └── *.pdf                      # Acts, rules and child-rights documents
│
├── outputs/
│   ├── benchmark_dataset.json     # Benchmark queries and annotations
│   ├── evaluation_summary.json    # Latest benchmark result
│   ├── *embeddings_cache.pkl      # Model-specific embedding cache
│   └── paper_figures/             # Publication-ready figures and table
│
├── notebooks/                     # Exploratory notebooks
├── presentations/                 # Presentation material and scripts
└── old_results/                   # Historical experimental outputs
```

---

## 4. Core Modules

### `app.py` — Application Entry Point

Responsible for the interactive workflow:

- Load or create the embedding cache.
- Present retrieval options.
- Accept user questions.
- Run child-law domain validation.
- Invoke the selected retrieval method.
- Display retrieved statutory provisions and scores.
- Generate a grounded answer.
- Maintain short-term conversation history.

The application can expose the experimental corrective retrieval method, but the research benchmark's controlled comparison contains four methods.

### `config.py` — Configuration and Global Settings

Centralizes:

- Project paths.
- Ollama host and timeout.
- LLM and embedding model names.
- Embedding batch size and request pacing.
- Chunking parameters.
- Retrieval constants such as `DEFAULT_TOP_K`, `RRF_K`, `BM25_K1`, and `BM25_B`.
- Child-law keywords and state mappings.
- Global retrieval/citation bookkeeping.

The current local model configuration is:

```text
LLM_MODEL       = llama3.2
EMBEDDING_MODEL = bge-m3
```

Embedding caches are model-specific to avoid reusing vectors produced by a different embedding model.

### `document_processing.py` — Document Loading and Chunking

Handles legal-corpus preparation:

- Discover source PDFs/text files.
- Extract document text.
- Attach source metadata.
- Split documents into structured chunks.
- Preserve section/article/rule information where available.
- Support configured state-specific sources.

### `embeddings.py` — Embedding Generation and Cache

Connects document chunks and queries to the local Ollama embedding API.

Responsibilities include:

- Query embedding.
- Batched corpus embedding.
- Embedding validation.
- Cache saving/loading.
- Cache existence checks.

The research embedding model is **BGE-M3**.

### `retrieval.py` — Retrieval Engine

Implements the retrieval strategies used by the application and benchmark.

#### 1. BM25 — Lexical Baseline

`retrieve_bm25(question, chunks, top_k)`

Uses BM25 Okapi over tokenized legal text. It provides a strong lexical baseline for statutory terms, phrases and explicit section numbers.

#### 2. Dense Retrieval — Semantic Baseline

`retrieve_dense(question, chunks, top_k)`

Embeds the query with BGE-M3 and ranks chunks by cosine similarity against cached chunk embeddings.

#### 3. Hybrid RRF Retrieval

`retrieve_hybrid_rrf(question, chunks, top_k)`

Combines BM25 and dense rankings using Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

where the configured `RRF_K` controls the rank contribution.

#### 4. Hybrid + Local Rerank

`retrieve_reranked(question, chunks, top_k)`

Builds a larger Hybrid RRF candidate pool and applies a local scoring/reranking stage using retrieval signals such as semantic similarity, lexical overlap, rank information, and legal-provision matching.

#### Experimental: Corrective Retrieval

`retrieve_crag_hyde(question, chunks, top_k)`

Contains a corrective retrieval pathway intended to detect weak initial retrieval and perform an additional retrieval/correction stage. It is retained as an experimental capability but is excluded from the primary paper comparison because its current benchmark behavior does not provide a sufficiently distinct result from Hybrid RRF.

---

## 5. `answer_generation.py` — Domain Validation and Grounded Generation

This module converts retrieved legal context into the final response.

### Domain validation

The module checks whether a query belongs to the child-law domain and can redirect unsupported queries before retrieval/generation.

### State-specific filtering

Questions referring to Tamil Nadu or Kerala can be used to prioritize/filter configured state-specific legal sources.

### Grounded generation

The prompt supplied to the local LLM includes:

- Act/source title.
- Section, Article or Rule identifier.
- Retrieved statutory text.
- User question.
- Grounding and citation instructions.

Generation is performed by **Llama 3.2 through Ollama**.

---

## 6. `evaluation.py` — Benchmark and Evaluation Framework

The evaluator provides the controlled research experiment independently of the interactive application.

### Benchmark

```text
35 total queries
├── 30 in-domain legal questions
└── 5 out-of-domain negative controls
```

The in-domain set is organized into six categories:

```text
PEN   = offence / penalty-related questions
PROC  = procedure-related questions
CNCP  = children in need of care and protection
CCL   = children in conflict with law
OFF   = child-law offence / related provisions
CONST = constitutional child-rights questions
```

### Retrieval ground truth

Each in-domain benchmark case contains annotated target statutory sections/provisions. The primary retrieval evaluation checks whether those target provisions occur in the retrieved top-k results.

### Primary metrics

**Hit@1** — relevant target appears at rank 1.

**Hit@3** — relevant target appears within the top 3.

**Hit@5** — relevant target appears within the top 5.

**MRR@5** — reciprocal rank of the first relevant result, averaged across the benchmark.

**Average Retrieval Latency** — average time required to return the top-k retrieval result.

### Guardrail evaluation

The five out-of-domain controls measure whether unsupported questions are correctly rejected by the child-law domain guardrail.

### Answer-level metrics

The evaluator contains additional answer-level calculations, but these are not primary research metrics. The current benchmark configuration generates/evaluates only a limited number of answers per method and does not use an LLM judge in the main run. Therefore context-support, gold-answer similarity and citation-accuracy measures are excluded from the primary paper table.

---

## 7. `generate_paper_figures.py` — Publication Outputs

This script reads the latest `outputs/evaluation_summary.json` and generates retrieval-focused paper figures.

```text
outputs/paper_figures/
├── figure_1_hit_rates.png
├── figure_2_mrr.png
├── figure_3_latency.png
├── figure_4_accuracy_latency_tradeoff.png
├── figure_5_category_hit3.png
└── table_1_main_comparison.png
```

The primary comparison table contains:

```text
Retrieval Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Avg. Latency
```

This keeps the main results aligned with the paper's retrieval-centered research question.

---

## 8. `law_data/` — Legal Corpus

Contains the source documents indexed by the system. The corpus includes configured central Indian child-law legislation/rules and child-rights material, with state-specific sources available where configured.

Source metadata is retained during ingestion so retrieved chunks can be associated with a source document and statutory provision.

---

## 9. `outputs/` — Runtime and Research Artifacts

```text
outputs/
├── benchmark_dataset.json
├── evaluation_summary.json
├── *embeddings_cache.pkl
└── paper_figures/
```

`old_results/` contains historical experiments and should not be mixed with the final benchmark results.

---

## 10. Reproducibility Rules

For a fair retrieval comparison:

1. Keep the corpus fixed across methods.
2. Keep the benchmark dataset fixed.
3. Use the same target-section annotations.
4. Keep BGE-M3 fixed as the embedding model.
5. Evaluate the same top-k range, with the main cutoff at 5.
6. Run latency measurement in the same environment.
7. Rerun the complete benchmark whenever retrieval code changes.
8. Generate paper figures only from the corresponding benchmark result file.

Changing the embedding model requires rebuilding the embedding cache and rerunning the benchmark.

---

## 11. Adding a Retrieval Method

1. Add a function to `retrieval.py` using `(question, chunks, top_k)`.
2. Expose it from `app.py` if it should be available interactively.
3. Add it to `evaluation.py` only if it is intended for controlled research comparison.
4. Document its algorithm and hyperparameters here.
5. Rerun the benchmark.
6. Add it to `generate_paper_figures.py` only if it becomes part of the primary paper method set.

---

## 12. Design Principles

### Separation of Concerns

Configuration, processing, embeddings, retrieval, generation and evaluation are independently maintained.

### Legal Provenance

Retrieved chunks retain source and provision metadata so generated answers can be traced to legal context.

### Local Execution

The core generation and embedding stages use local Ollama models rather than cloud inference APIs.

### Reproducibility

The corpus, benchmark, model configuration, cache and evaluation output are explicit artifacts.

### Research Transparency

The paper should report only metrics that are supported by the actual benchmark design.

---

## 13. Academic Scope

The project is best described as a **comparative evaluation of retrieval strategies for local child-law RAG**. The legal assistant demonstrates the practical application, while the controlled retrieval benchmark forms the main empirical research component.

The primary comparison follows this progression:

```text
BM25
  ↓
Dense (BGE-M3)
  ↓
Hybrid (RRF)
  ↓
Hybrid + Local Rerank
```

This provides a clear experimental progression from lexical retrieval to semantic retrieval, rank fusion, and reranking.

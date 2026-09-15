# Design and Development of an AI-Powered Legal Assistant for Child Protection and Rights Awareness

A local, domain-grounded Retrieval-Augmented Generation (RAG) system for answering questions about Indian child protection and child-rights legislation. The project also serves as a research framework for empirically comparing retrieval strategies on a child-law benchmark.

## Project Scope

The system focuses on central Indian child-law sources, including:

- Protection of Children from Sexual Offences (POCSO) Act, 2012
- POCSO Rules, 2020
- Juvenile Justice (Care and Protection of Children) Act, 2015
- Juvenile Justice Model Rules, 2017
- Relevant child-rights provisions of the Constitution of India

State-specific document support is also present in the codebase for Tamil Nadu and Kerala where applicable.

## Research Focus

The main research objective is a controlled comparison of retrieval methods for local RAG-based child-law question answering.

The final paper comparison uses four genuinely distinct retrieval strategies:

1. **BM25 (Lexical)** — lexical matching based on statutory terms, phrases, and section numbers.
2. **Dense Retrieval (BGE-M3)** — semantic retrieval using locally generated vector embeddings and cosine similarity.
3. **Hybrid Retrieval (RRF)** — combines dense and BM25 rankings using Reciprocal Rank Fusion.
4. **Hybrid + Local Rerank** — retrieves a larger hybrid candidate set and applies a local scoring/reranking stage before returning the final top-k results.

The codebase also contains a corrective retrieval implementation. It is retained as an experimental capability but is excluded from the main paper comparison because, on the current benchmark, it does not provide a sufficiently distinct retrieval outcome from Hybrid RRF.

## System Architecture

```text
Legal PDFs / Text Files
        |
        v
Document Loading
        |
        v
Structure-Aware Chunking
        |
        v
Local Embedding Generation (Ollama + BGE-M3)
        |
        v
Embedding Cache
        |
        v
+-------------------------------+
| Retrieval Engine              |
| BM25 | Dense | Hybrid | Rerank|
+-------------------------------+
        |
        v
Top-k Legal Chunks + Sources
        |
        v
Domain-Checked Grounded Prompt
        |
        v
Local LLM (Ollama + Llama 3.2)
        |
        v
Answer + Statutory Citations
```

## Key Engineering Features

### Fully Local Models

The application uses Ollama for both generation and embeddings:

- **LLM:** `llama3.2`
- **Embedding model:** `bge-m3`
- **Ollama API:** local `http://localhost:11434`

No cloud LLM or embedding API is required for the core RAG pipeline.

### Structure-Aware Legal Chunking

Legal documents are processed into chunks while preserving statutory boundaries and metadata such as the source act, section/article/rule identifier, and chunk ID. This allows retrieval results to retain a clear legal provenance.

### Embedding Cache

Generated embeddings are cached under `outputs/` so that subsequent runs do not need to recompute the complete corpus. The cache filename is model-specific to avoid accidentally mixing embeddings from different embedding models.

### Domain Guardrail

Before retrieval/generation, the application checks whether the query belongs to the child-law domain. Out-of-domain questions can be rejected or redirected instead of being passed directly to the legal RAG pipeline.

### Citation-Grounded Generation

The answer generator receives the retrieved legal context together with source metadata and is instructed to ground substantive statements in the provided context and reference the relevant statutory provisions.

## Project Structure

```text
child_law_rag/
├── app.py                         # Interactive local legal assistant
├── config.py                      # Paths, models, Ollama settings and hyperparameters
├── document_processing.py         # PDF loading and structure-aware chunking
├── embeddings.py                  # Local embedding generation and cache management
├── retrieval.py                   # BM25, Dense, Hybrid RRF, reranking and corrective retrieval
├── answer_generation.py           # Domain validation and grounded answer generation
├── evaluation.py                  # Benchmark execution and retrieval metrics
├── generate_paper_figures.py      # Publication figures and main comparison table
├── test_ollama.py                 # Ollama connectivity/model smoke test
├── requirements.txt               # Python dependencies
├── .env.example                   # Example local configuration
├── README.md                      # Project documentation
├── STRUCTURE.md                   # Detailed architecture and module documentation
├── RESEARCH_PAPER.md              # Research-paper manuscript
├── diagram.md                     # Architecture diagram description
├── diagramv2.md                   # Alternate architecture diagram description
│
├── law_data/                      # Legal source documents
│   ├── *.pdf                      # Acts, rules and child-rights source documents
│   └── ...
│
├── outputs/
│   ├── benchmark_dataset.json     # 35-query benchmark (30 in-domain + 5 negative controls)
│   ├── *embeddings_cache.pkl      # Local embedding cache
│   ├── evaluation_summary.json    # Benchmark results
│   └── paper_figures/             # Publication-ready figures/table
│
├── notebooks/                     # Exploratory notebooks
├── presentations/                 # Project presentation material
└── old_results/                   # Earlier experimental outputs
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ollama Setup

Install and start Ollama, then make sure the required models are available:

```bash
ollama pull llama3.2
ollama pull bge-m3
ollama list
```

The project reads model names and Ollama settings from `.env` when provided. Typical settings are:

```env
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_MODEL=bge-m3
BATCH_EMBED_SIZE=8
```

A smaller embedding batch size can be useful for stable local processing on systems with limited memory.

## Build / Rebuild the Embedding Index

When the legal corpus or embedding model changes, rebuild the embedding cache. The resulting cache is stored in `outputs/` using a model-specific name.

The normal pipeline is:

```text
load documents
    -> chunk documents
    -> generate BGE-M3 embeddings
    -> validate/cache embeddings
```

## Run the Assistant

```bash
python app.py
```

The interactive application allows the user to select a retrieval strategy, submit child-law questions, inspect retrieved statutory provisions, and receive a grounded answer generated by the local Llama 3.2 model.

## Research Benchmark

The benchmark currently contains:

- **30 in-domain legal questions**
- **5 out-of-domain negative controls**
- Six in-domain categories: PEN, PROC, CNCP, CCL, OFF and CONST

Retrieval is evaluated against annotated target statutory sections rather than relying on generated-answer similarity as the primary ground truth.

### Primary Evaluation Metrics

The main paper evaluation uses:

- **Hit@1** — whether the relevant target provision appears at rank 1.
- **Hit@3** — whether it appears in the top 3 results.
- **Hit@5** — whether it appears in the top 5 results.
- **MRR@5** — mean reciprocal rank of the first relevant result within the top 5.
- **Average Retrieval Latency** — average time needed to produce the retrieved top-k context.

The out-of-domain controls additionally measure the accuracy of the child-law domain guardrail.

### Run the Benchmark

From the project root:

```bash
python -c "from embeddings import load_embeddings_cache; from evaluation import run_comprehensive_benchmark; chunks = load_embeddings_cache(); run_comprehensive_benchmark(chunks)"
```

The benchmark writes:

```text
outputs/evaluation_summary.json
```

and generates the project's evaluation artifacts.

## Generate Paper Figures and Main Table

The separate figure-generation script creates the final retrieval-only figures and the main comparison table:

```bash
python generate_paper_figures.py
```

Generated files:

```text
outputs/paper_figures/
├── figure_1_hit_rates.png
├── figure_2_mrr.png
├── figure_3_latency.png
├── figure_4_accuracy_latency_tradeoff.png
├── figure_5_category_hit3.png
└── table_1_main_comparison.png
```

The main comparison table contains only:

```text
Retrieval Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Avg. Latency
```

Answer-level exploratory measures such as context support, gold-answer similarity, and citation accuracy are not used in the main comparison because the current benchmark does not generate/judge answers for the complete query set.

## Reproducibility Notes

- Retrieval experiments should use the same corpus, benchmark dataset, chunking configuration, embedding cache, and top-k setting.
- Changing the embedding model requires rebuilding the embedding cache.
- Changing retrieval code requires rerunning the benchmark before reporting final results.
- The four primary paper methods should be treated as the controlled comparison set.
- Older outputs under `old_results/` are retained only as historical experiments and should not be mixed with the final benchmark results.

## Academic Paper

The manuscript is maintained in:

```text
RESEARCH_PAPER.md
```

The paper should describe the work primarily as a **comparative retrieval evaluation for local child-law RAG**, rather than as a claim of comprehensive legal-answer accuracy.

## Disclaimer

This project is intended for legal information, research, and educational purposes. It is not a substitute for formal legal advice, professional legal representation, or official legal services.
'''

cat > /mnt/data/STRUCTURE_updated.md <<'EOF'
# Project Structure and Architecture

## 1. Overview

`child_law_rag` is a modular local Retrieval-Augmented Generation (RAG) system for Indian child-law question answering. It is organized so that document processing, embedding generation, retrieval, answer generation, and evaluation can be developed and tested independently.

The repository has two related purposes:

1. **Application:** provide a locally running child-law legal assistant using Ollama.
2. **Research:** compare retrieval strategies under a fixed child-law benchmark.

The final research comparison uses four primary retrieval methods: BM25, Dense BGE-M3, Hybrid RRF, and Hybrid + Local Rerank.

---

## 2. High-Level Data Flow

```text
                 Legal Source Documents
                          |
                          v
                document_processing.py
                          |
                          v
                 Structured Chunks
                          |
                          v
                    embeddings.py
                          |
                 BGE-M3 Embeddings
                          |
                          v
                 outputs/*cache.pkl
                          |
                          v
                    retrieval.py
                          |
       +------------------+------------------+
       |                  |                  |
      BM25              Dense           Hybrid / Rerank
       |                  |                  |
       +------------------+------------------+
                          |
                          v
                 Retrieved Top-k Chunks
                          |
                          v
                  answer_generation.py
                          |
                          v
                    Llama 3.2
                          |
                          v
                Grounded Answer + Sources

                    Parallel research path
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
              Paper Figures + Table
```

---

## 3. Core Modules

### `app.py` — Application Entry Point

Responsible for the interactive assistant workflow.

Responsibilities:

- Load or build the embedding cache.
- Present the retrieval-strategy menu.
- Accept legal questions from the user.
- Run the child-law domain guardrail.
- Invoke the selected retrieval method.
- Display retrieved statutory provisions and scores.
- Generate a grounded legal response.
- Maintain short-term conversation history.

The application currently exposes the retrieval implementations directly, including the experimental corrective method, while the research benchmark uses the four-method controlled comparison set.

---

### `config.py` — Configuration and Global Settings

Central location for:

- Project paths.
- Ollama host and timeout.
- LLM and embedding model names.
- Embedding batch size and pacing.
- Chunking defaults.
- Retrieval constants such as `DEFAULT_TOP_K`, `RRF_K`, `BM25_K1`, and `BM25_B`.
- Domain keywords.
- Statute manifest.
- Global retrieval/citation bookkeeping.

Current local model configuration is:

```text
LLM_MODEL        = llama3.2
EMBEDDING_MODEL  = bge-m3
```

The embedding cache is model-specific so that a cache created for one embedding model is not silently reused for another.

---

### `document_processing.py` — Document Ingestion and Chunking

Handles preparation of the legal corpus.

Responsibilities:

- Discover legal source files.
- Extract text from PDFs.
- Attach document metadata.
- Split statutes into structured chunks.
- Preserve section/article/rule information where available.
- Support state-specific source handling.

The goal is to keep legal provisions structurally meaningful rather than treating the corpus as an undifferentiated block of text.

---

### `embeddings.py` — Embedding Generation and Cache

Provides the interface between the legal chunks and Ollama's local embedding API.

Responsibilities:

- Embed individual queries.
- Embed corpus chunks in batches.
- Validate embeddings before they are stored.
- Save embeddings to disk.
- Load the existing model-specific cache.
- Detect whether a usable cache exists.

The production/research embedding model is **BGE-M3**.

---

### `retrieval.py` — Retrieval Engine

Contains the retrieval implementations used by the application and benchmark.

#### 1. BM25 — Lexical Baseline

`retrieve_bm25(question, chunks, top_k)`

Uses BM25 Okapi scoring over tokenized legal text. This method is particularly suited to exact statutory terminology, named provisions, and section-number matching.

#### 2. Dense Retrieval — Semantic Baseline

`retrieve_dense(question, chunks, top_k)`

Embeds the query with BGE-M3 and compares it with cached chunk embeddings using cosine similarity.

#### 3. Hybrid RRF Retrieval

`retrieve_hybrid_rrf(question, chunks, top_k)`

Combines the ranked outputs of BM25 and dense retrieval through Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

with the standard constant configured in `RRF_K`.

The purpose is to combine lexical exactness with semantic matching without requiring manually tuned score-scale weights.

#### 4. Hybrid + Local Rerank

`retrieve_reranked(question, chunks, top_k)`

Uses Hybrid RRF to create a larger candidate pool and then applies a local reranking stage using retrieval features such as semantic similarity, lexical overlap, rank information, and legal-provision matching signals.

This is the fourth and strongest retrieval configuration in the primary research comparison.

#### Experimental: Corrective Retrieval

`retrieve_crag_hyde(question, chunks, top_k)`

The repository contains a corrective retrieval implementation intended to identify weak initial retrieval and perform a secondary retrieval/correction step. It is retained as an experimental capability but is **not part of the final four-method paper table** because its current benchmark behavior does not provide a sufficiently distinct comparison from Hybrid RRF.

---

## 4. `answer_generation.py` — Domain Validation and Grounded Generation

Responsible for converting retrieved legal context into the final user-facing response.

### Domain validation

The module checks whether a query belongs to the supported child-law domain. Out-of-domain queries can be rejected or redirected before retrieval and answer generation.

### State-specific filtering

Where relevant, questions mentioning Tamil Nadu or Kerala can be used to restrict retrieval to state-specific sources when those documents are available.

### Grounded answer generation

The module constructs a prompt containing:

- Source statute name.
- Section/article/rule identifier.
- Retrieved statutory text.
- The user's question.
- Grounding and citation instructions.

The response is generated with the local Ollama LLM (`llama3.2`).

---

## 5. `evaluation.py` — Benchmark and Evaluation Framework

The evaluation module separates retrieval evaluation from the application UI.

### Benchmark structure

Current benchmark:

```text
35 total queries
├── 30 in-domain legal questions
└── 5 out-of-domain negative controls
```

The in-domain questions are grouped into:

```text
PEN   = Penal / offence-related
PROC  = Procedure
CNCP  = Children in Need of Care and Protection
CCL   = Children in Conflict with Law
OFF   = Child-law related offences / related provisions
CONST = Constitutional child-rights questions
```

### Retrieval ground truth

For the main experiment, correctness is judged by whether the retrieved results contain the annotated **target statutory sections/provisions** associated with each benchmark question.

The benchmark therefore measures retrieval effectiveness directly rather than treating generated-answer similarity as the primary ground truth.

### Primary metrics

#### Hit@1

Fraction of benchmark queries for which a target provision occurs at rank 1.

#### Hit@3

Fraction of benchmark queries for which a target provision occurs within the top 3 retrieved results.

#### Hit@5

Fraction of benchmark queries for which a target provision occurs within the top 5 retrieved results.

#### MRR@5

Mean reciprocal rank of the first relevant result, considering the top five positions.

#### Average Retrieval Latency

Mean elapsed time from the retrieval call to the returned top-k list.

### Domain Guardrail Metric

The five out-of-domain negative controls are used to measure whether unsupported questions are correctly rejected. This metric is reported separately from retrieval effectiveness.

### Answer-level exploratory metrics

The evaluator contains optional answer-level measures such as faithfulness/relevance-style scores and citation accuracy. These are not used in the primary paper comparison because the current benchmark configuration does not evaluate generated answers for the full 30-query in-domain set and the LLM judge is disabled.

---

## 6. `generate_paper_figures.py` — Publication Artifacts

Generates the final retrieval-focused visuals from `outputs/evaluation_summary.json`.

The final outputs are:

```text
outputs/paper_figures/
├── figure_1_hit_rates.png
├── figure_2_mrr.png
├── figure_3_latency.png
├── figure_4_accuracy_latency_tradeoff.png
├── figure_5_category_hit3.png
└── table_1_main_comparison.png
```

### Main comparison table

The paper table contains:

```text
Retrieval Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Avg. Latency
```

The table intentionally excludes context-support, gold-answer similarity, and citation-accuracy columns so that the main comparison remains aligned with the retrieval-centered research question.

---

## 7. Data and Output Directories

### `law_data/`

Contains the source legal corpus used for ingestion and retrieval.

Typical source categories include:

- POCSO legislation and rules.
- Juvenile Justice legislation and rules.
- Constitutional text.
- Additional child-protection/right documents where configured.
- State-specific Tamil Nadu/Kerala documents where used.

### `outputs/`

Stores generated runtime and research artifacts:

```text
outputs/
├── benchmark_dataset.json
├── evaluation_summary.json
├── *embeddings_cache.pkl
└── paper_figures/
```

Historical experiments are retained under `old_results/` and should not be mixed with the final benchmark.

---

## 8. Research Reproducibility

For a reproducible comparison:

1. Use the same corpus for all methods.
2. Use the same benchmark query set.
3. Use the same target-section annotations.
4. Keep the embedding model fixed at BGE-M3.
5. Keep the top-k evaluation cutoff fixed at 5.
6. Measure latency for each retrieval call in the same environment.
7. Rerun the complete benchmark after changing retrieval code.
8. Generate paper figures only from the resulting `evaluation_summary.json`.

A new embedding model requires a new embedding cache and a new benchmark run.

---

## 9. Adding a New Retrieval Method

To add a retrieval strategy:

1. Implement a function in `retrieval.py` with the standard `(question, chunks, top_k)` interface.
2. Import it in `app.py` if it should be exposed in the interactive assistant.
3. Add it to the benchmark method dictionary in `evaluation.py` only when it is intended to be part of the controlled evaluation.
4. Document the method and its hyperparameters here.
5. Rerun the complete benchmark before comparing results.
6. Update `generate_paper_figures.py` if the new method is accepted into the primary paper comparison.

---

## 10. Configuration Summary

The main configuration is centralized in `config.py` and can be overridden through `.env`.

Conceptually:

```text
Ollama host        -> localhost:11434
LLM                 -> llama3.2
Embeddings          -> bge-m3
Chunk size          -> configured in config.py
Chunk overlap       -> configured in config.py
Default top-k       -> configured in config.py
RRF constant        -> RRF_K
BM25 parameters     -> BM25_K1 / BM25_B
Embedding batch     -> BATCH_EMBED_SIZE
```

Exact values should be taken from the active `config.py` rather than copied into other modules.

---

## 11. Design Principles

### Separation of Concerns

Each major responsibility is isolated in its own module.

### Reproducibility

Model configuration, caching, benchmark data, and evaluation are kept explicit so results can be regenerated.

### Legal Provenance

Retrieved chunks retain source and provision metadata so answers can be traced to specific statutory context.

### Local Execution

The core LLM and embedding components run through Ollama rather than requiring cloud inference APIs.

### Research Transparency

The primary paper claims are limited to metrics actually supported by the benchmark design.

---

## 12. Academic Scope

The project should be described in the paper as a **comparative evaluation of retrieval strategies for local child-law RAG**. The application component demonstrates a practical use case, while the controlled benchmark provides the primary empirical contribution.

The current four-method comparison is:

```text
BM25
  ↓
Dense (BGE-M3)
  ↓
Hybrid (RRF)
  ↓
Hybrid + Local Rerank
```

This progression provides a clear experimental narrative from lexical retrieval through semantic, fused, and reranked retrieval.

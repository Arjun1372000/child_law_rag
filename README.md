# Design and Development of an AI-Powered Legal Assistant for Child Protection and Rights Awareness

A specialized, domain-grounded legal assistant and empirical research framework designed to provide accurate, citation-verified legal information on Central Indian Child Protection Laws (POCSO Act 2012, Juvenile Justice Act 2015, and the Constitution of India).

---

## 🌟 Key Research & Engineering Features

- **Cloud API Architecture**: Powered by Google Gemini (`gemini-3.6-flash` and `gemini-embedding-001`) via official `google-genai` SDK, eliminating local GPU/hardware constraints.
- **Dual-Stage Domain Guardrail**:
  - *Stage 1 (Lexical Fast-Filter)*: Instant matching of high-signal statutory child protection terms.
  - *Stage 2 (Semantic Zero-Shot Triage)*: Accurately classifies and politely redirects out-of-domain queries (e.g., corporate tax, adult criminal disputes, general non-legal chatter).
- **Structure-Aware Statutory Chunking**: Preserves Section, Article, and Rule legislative boundaries (Act $\rightarrow$ Chapter $\rightarrow$ Section $\rightarrow$ Clause) rather than naive character slicing.
- **5 Comparative Retrieval Strategies**:
  1. **BM25 Okapi**: Exact statutory term and section number matching.
  2. **Dense Vector Embeddings**: 3072-dimensional semantic representation with cosine similarity.
  3. **Hybrid Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum \frac{1}{60 + rank_i(d)}$ eliminating arbitrary score weight tuning.
  4. **Two-Stage Re-ranking**: Hybrid candidate pooling followed by pointwise LLM cross-scoring for peak precision.
  5. **Corrective RAG (HyDE)**: Hypothetical Document Embeddings bridging layperson language with formal statutory drafting style.
- **ChildLaw-QA Benchmark & Scientific Evaluation Engine**:
  - 30 annotated test cases across 6 legal categories.
  - Automated calculation of **Hit@1, Hit@3, Hit@5, MRR@5, Faithfulness (Hallucination score), Citation Accuracy**, and latency.
  - Export of publication-ready **LaTeX tables** (`outputs/benchmark_comparison_table.tex`) and visual comparison plots.
- **Complete Academic Research Paper**: Pre-drafted academic manuscript ready for review in [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md).

---

## 🛠️ Technology Stack

- **LLM & Embeddings**: Google Gemini API (`gemini-3.6-flash`, `gemini-embedding-001`)
- **Document Processing**: `pypdf`, regex-based statutory boundary chunker
- **Information Retrieval**: `rank-bm25`, NumPy, Gemini vector embeddings
- **Evaluation & Visuals**: Matplotlib, Seaborn, JSON, LaTeX table generator
- **Resilience**: `tenacity` exponential backoff, persistent disk checkpointing, and cache

---

## 📁 Project Structure

```
child_law_rag/
├── app.py                      # Upgraded interactive console assistant
├── config.py                   # Central configurations & rate-limit pacing
├── document_processing.py      # Structure-aware statutory chunker
├── embeddings.py               # Gemini batch embedding with checkpointing
├── retrieval.py                # 5 comparative retrieval strategies (BM25, Dense, RRF, Rerank, HyDE)
├── answer_generation.py        # Dual-stage domain guardrail & citation-grounded generator
├── evaluation.py               # Scientific IR & generation benchmarking engine
├── requirements.txt            # Project dependencies
├── RESEARCH_PAPER.md           # Full academic research paper manuscript
├── law_data/                   # Central Indian Child Protection Bare Acts & Rules
│   ├── POCSOact_pt1.pdf        # Protection of Children from Sexual Offences Act, 2012
│   ├── POCSOrules.pdf          # POCSO Rules, 2020
│   ├── jjact2015.pdf           # Juvenile Justice Act, 2015
│   ├── juvenile_justice_rules_2017.pdf # Juvenile Justice Model Rules, 2017
│   └── constitution_english.pdf # Indian Constitution (Child Rights provisions)
└── outputs/                    # Benchmarks, checkpoints, cache & research artifacts
    ├── benchmark_dataset.json  # ChildLaw-QA benchmark
    ├── benchmark_comparison_table.md   # Benchmark Markdown table
    ├── benchmark_comparison_table.tex  # Benchmark LaTeX table
    └── evaluation_comparison_charts.png # High-res comparison plots
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_LLM_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### 3. Run the Interactive Assistant
```bash
python app.py
```
From the interactive menu, you can:
- Query using any of the 5 comparative retrieval methods.
- View section-level statutory source citations.
- Select Option 6 to run the full scientific benchmark.

---

## 📊 Evaluation & Research Benchmark

Run the evaluation engine programmatically:
```bash
python -c "from embeddings import load_embeddings_cache; from evaluation import run_comprehensive_benchmark; chunks = load_embeddings_cache(); run_comprehensive_benchmark(chunks)"
```
This automatically updates:
- `outputs/benchmark_comparison_table.md`
- `outputs/benchmark_comparison_table.tex`
- `outputs/evaluation_comparison_charts.png`

---

## 📝 Academic Paper
Read the complete research paper detailing methodology, mathematics, empirical evaluation, and ethical considerations in [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md).

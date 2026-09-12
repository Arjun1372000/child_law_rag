# Project Structure Documentation

## Overview
The Child Law Legal Assistant has been refactored into a modular, organized structure. The main application logic is now separated into focused modules, making the codebase more maintainable, testable, and scalable.

## Module Organization

### `app.py` - Main Application
The entry point of the program. Handles:
- User interaction and conversation loop
- Orchestrating different retrieval methods
- Managing conversation history
- Display of results and sources

**Key Functions:**
- `main()` - Main program entry point with interactive menu
- Imports and uses functions from all other modules

---

### `config.py` - Configuration & Constants
Central configuration file containing:
- OpenAI client setup (Ollama connection)
- Global state variables (LAST_RETRIEVAL_SCORES, CITATION_MAP)
- Default parameters and hyperparameters
- Model names and cache settings
- Domain-specific keywords and state mappings

**Key Variables:**
- `client` - OpenAI client for API calls
- `EMBEDDING_MODEL` - Embedding model name
- `LLM_MODEL` - Language model name
- `DENSE_WEIGHT`, `BM25_WEIGHT` - Hybrid retrieval weights
- `CHILD_LAW_KEYWORDS` - Domain keywords
- `STATE_KEYWORDS` - State-specific keywords

---

### `document_processing.py` - Document Loading & Chunking
Handles document corpus preparation:
- Loading documents from folders (PDF and text files)
- State-specific document loading (Tamil Nadu, Kerala)
- Document chunking with configurable size and overlap

**Key Functions:**
- `load_documents(folder, include_states)` - Load documents
- `chunk_documents(documents, chunk_size, overlap)` - Split documents into chunks

---

### `embeddings.py` - Embedding Management
Manages embedding creation and caching:
- Creating embeddings for document chunks
- Saving embeddings to disk for reuse
- Loading cached embeddings
- Checking cache existence

**Key Functions:**
- `embed_chunks(chunks)` - Generate embeddings
- `save_embeddings_cache(chunks, cache_file)` - Save to disk
- `load_embeddings_cache(cache_file)` - Load from cache
- `embeddings_cache_exists(cache_file)` - Check cache

---

### `retrieval.py` - Retrieval Methods
Implements four different retrieval strategies:

1. **Dense Retrieval** - Uses embedding similarity
   - `retrieve_dense(question, chunks, top_k)`

2. **Hybrid Retrieval** - Combines dense + BM25 (70% dense, 30% BM25)
   - `retrieve_hybrid(question, chunks, top_k)`
   - Uses `SimpleBM25` fallback if rank_bm25 unavailable

3. **Multi-Query Retrieval** - Generates query variants and aggregates results
   - `retrieve_multiquery(question, chunks, top_k)`
   - `generate_query_variants(question)` - Generate alternative queries

4. **Corrective Retrieval** - Fallback method with quality checks
   - `retrieve_corrective(question, chunks, top_k)`

**Helper Classes:**
- `SimpleBM25` - Fallback BM25 implementation

**Utility Functions:**
- `get_bm25_scores(query, texts)` - Get BM25 scores

---

### `answer_generation.py` - Answer Processing
Generates and validates answers:
- Domain-specific question validation
- State-based chunk filtering
- Answer generation with citations
- Conversation history management

**Key Functions:**
- `is_child_law_question(question)` - Validate domain
- `filter_chunks_for_state(question, chunks)` - Filter by state
- `generate_answer(question, retrieved_chunks, conversation_history)` - Generate answer

---

### `evaluation.py` - Evaluation & Visualization
Comprehensive evaluation framework:
- LLM-based answer relevance scoring
- Performance metrics (latency, coverage)
- Visualization graphs
- Metrics table generation

**Key Functions:**
- `evaluate_answer_relevance(question, answer, retrieved_chunks)` - Score 0.0-1.0
- `generate_evaluation_graphs(results)` - Create performance charts
- `generate_relevance_metrics_table(results)` - Create metrics table
- `run_evaluation(chunks)` - Full evaluation pipeline

---

## File Structure

```
child_law_project/
├── app.py                      # Main application entry point
├── config.py                   # Configuration & constants
├── document_processing.py      # Document loading & chunking
├── embeddings.py               # Embedding management
├── retrieval.py                # Retrieval methods
├── answer_generation.py        # Answer generation
├── evaluation.py               # Evaluation & visualization
├── law_data/                   # Document storage
│   ├── *.pdf                   # Base documents
│   ├── tamil_nadu/             # Tamil Nadu specific docs
│   └── kerala/                 # Kerala specific docs
├── outputs/                    # Generated outputs
│   ├── embeddings_cache.pkl    # Cached embeddings
│   ├── evaluation_results.json # Evaluation results
│   ├── evaluation_graphs.png   # Performance graphs
│   └── relevance_metrics.html  # Metrics table
└── STRUCTURE.md               # This file
```

---

## Key Design Principles

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- Configuration is centralized
- Document processing is isolated
- Retrieval methods are modular
- Evaluation is independent

### 2. **Reusability**
Modules can be imported and used independently:
```python
from retrieval import retrieve_dense, retrieve_hybrid
from embeddings import load_embeddings_cache
```

### 3. **Maintainability**
- Clear function responsibilities with docstrings
- Configurable parameters in config.py
- Easy to modify behavior or add new methods

### 4. **Scalability**
- Easy to add new retrieval methods
- New evaluation metrics can be added
- Support for additional state-specific documents

---

## Usage

### Running the Main Application
```bash
python app.py
```

### Importing Modules
```python
from config import client, EMBEDDING_MODEL
from retrieval import retrieve_hybrid
from embeddings import load_embeddings_cache
from evaluation import run_evaluation
```

### Running Evaluation
Within the interactive menu, select option "5" to run comprehensive evaluation.

---

## Configuration

To modify default parameters, edit `config.py`:

```python
# Retrieval parameters
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 150

# Model selection
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Hybrid weights
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
```

---

## Data Flow

```
Documents
   ↓
load_documents() → Document dicts
   ↓
chunk_documents() → Chunks
   ↓
embed_chunks() → Chunks with embeddings
   ↓ (cached in outputs/embeddings_cache.pkl)
   ↓
retrieve_*() → Retrieved chunks → generate_answer() → Answer
   ↓
evaluate_answer_relevance() → Relevance score
   ↓
Visualization & metrics
```

---

## Adding New Features

### Adding a New Retrieval Method
1. Create function in `retrieval.py`
2. Follow existing patterns for signature
3. Update loop in `app.py` main() function
4. Add to methods dict in `run_evaluation()`

### Adding New Configuration
1. Add to `config.py`
2. Import in required modules
3. Use consistently

### Extending Evaluation
1. Add function to `evaluation.py`
2. Call from `run_evaluation()` in evaluation.py
3. Update visualization if needed

---

## Dependencies

Key external libraries:
- `openai` - API client
- `pypdf` - PDF processing
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualization
- `rank_bm25` - BM25 scoring (optional, with fallback)

---

## Notes

- The application uses Ollama for local LLM inference
- Embeddings are cached locally for performance
- Conversation history is kept to the last 3 exchanges
- Domain validation ensures questions are about child law
- State-specific filtering improves relevance for location-based queries

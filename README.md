# Child Law Legal Assistant

A specialized legal assistant powered by LLMs (Large Language Models) designed to provide answers on child protection laws, child rights, and related legal matters in India. The system supports multiple retrieval strategies to ensure accurate and contextually relevant responses.

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that combines document retrieval with LLM-based answer generation. It specializes in child law matters and supports state-specific legal documents (Tamil Nadu, Kerala, etc.).

## ✨ Key Features

- **Multi-Strategy Retrieval**: Four different retrieval methods to optimize answer quality
  - **Dense Retrieval**: Semantic similarity using embeddings
  - **Hybrid Retrieval**: Combines embedding-based and BM25 keyword matching (70/30 split)
  - **Multi-Query Retrieval**: Generates query variants for comprehensive coverage
  - **Corrective Retrieval**: Fallback with quality validation mechanisms

- **State-Specific Support**: Load and query documents specific to different states (Tamil Nadu, Kerala)
- **Smart Conversation Management**: Maintains conversation history for contextual responses
- **Embedding Caching**: Efficient caching of embeddings to reduce computation time
- **Domain Validation**: Automatically validates if questions are related to child law
- **Comprehensive Evaluation**: Built-in evaluation framework with relevance scoring and performance metrics

## 🛠️ Technology Stack

- **LLM Framework**: OpenAI API (via Ollama local deployment)
- **Models**:
  - Embedding: `nomic-embed-text`
  - Language: `llama3.2`
- **Key Libraries**:
  - Document Processing: PyPDF2, text processing
  - Retrieval: Dense embedding similarity, BM25 keyword matching
  - Evaluation: LLM-based relevance scoring with visualization

## 📁 Project Structure

```
child_law_rag/
├── app.py                      # Main interactive application
├── config.py                   # Configuration & constants
├── document_processing.py      # Document loading & chunking
├── embeddings.py               # Embedding creation & caching
├── retrieval.py                # Retrieval strategies
├── answer_generation.py        # Answer generation & filtering
├── evaluation.py               # Metrics & visualization
├── requirements.txt            # Project dependencies
├── law_data/                   # Document storage
│   ├── *.pdf                   # Legal Bare Acts & Rules (POCSO, JJ Act, Constitution)
│   ├── tamil_nadu/             # Tamil Nadu specific laws
│   └── kerala/                 # Kerala specific laws
├── outputs/                    # Generated outputs, evaluation graphs & cache
├── notebooks/                  # Prototyping and benchmark notebooks
│   ├── law.ipynb
│   └── lawV2.ipynb
├── presentations/              # Academic review presentations & generator scripts
│   ├── Child_Law_Architecture.pptx
│   ├── Final_Review2_ChildLawAssistant.pptx
│   ├── create_architecture_ppt.py
│   └── ppt.py
└── webapp/                     # Web application interface
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama running locally on `http://localhost:11434`
- Required packages (see requirements.txt)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```

The interactive menu allows you to:
1. Ask legal questions about child law
2. Try different retrieval methods
3. View citation sources
4. Run comprehensive evaluation

## 📊 Retrieval Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Dense** | Semantic similarity from embeddings | General queries |
| **Hybrid** | Combines embeddings + keyword matching | Balanced accuracy |
| **Multi-Query** | Multiple query variants aggregated | Complex questions |
| **Corrective** | Quality-validated fallback retrieval | Edge cases |

## ⚙️ Configuration

Edit `config.py` to customize:
- Retrieval parameters (`DEFAULT_TOP_K`, `DEFAULT_CHUNK_SIZE`)
- Model names (embedding & LLM models)
- Retrieval weights (dense vs BM25)
- State-specific keywords

## 📈 Evaluation

The project includes comprehensive evaluation tools:
- **Relevance Scoring**: LLM-based assessment of answer quality (0.0-1.0)
- **Performance Metrics**: Latency, coverage, and accuracy tracking
- **Visualization**: Performance graphs and metrics tables
- **Reports**: JSON export of evaluation results

Run evaluation from the interactive menu or programmatically:
```python
from evaluation import run_evaluation
results = run_evaluation(chunks)
```

## 🔍 Domain Focus

Specializes in:
- Child Protection Laws (POCSO Act)
- Child Rights and Welfare
- Adoption and Guardianship Laws
- Child Labor Regulations
- Right to Education
- State-specific child law regulations

## 📝 License

See LICENSE file for details.

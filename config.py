import sys
from pathlib import Path
from openai import OpenAI

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Project base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "law_data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# OpenAI client configuration (using Ollama)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Global variables for tracking retrieval scores and citations
LAST_RETRIEVAL_SCORES = {}  # Store {(file, chunk_id): score}
CITATION_MAP = {}  # Store citation information for answers

# Default parameters for document processing
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 150

# Default parameters for retrieval
DEFAULT_TOP_K = 3

# Model names
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# BM25 hyperparameters
BM25_K1 = 1.5
BM25_B = 0.75

# Hybrid retrieval weights
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# Cache configuration
CACHE_FILE = "embeddings_cache.pkl"
CACHE_DIR = OUTPUTS_DIR

# Child law keywords
CHILD_LAW_KEYWORDS = [
    "child", "children", "minor", "pocso", "juvenile",
    "adoption", "guardian", "child labour", "child abuse",
    "right to education"
]

# State mappings
STATE_KEYWORDS = {
    "tamil_nadu": ["tamil nadu", "chennai"],
    "kerala": ["kerala", "kochi"]
}

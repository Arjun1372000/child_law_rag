# config.py
"""
Configuration, model initialization, and global settings for Child Law Legal Assistant.
Powered by Google Gemini API (gemini-3.6-flash and gemini-embedding-001).
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import APIError

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

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please add it to your .env file or environment variables."
    )

# Models
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

# Initialize official Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Free Tier Rate Pacing & Robustness
REQUEST_DELAY_SECONDS = 25.0 # Spacing between batches to stay under 100 RPM
BATCH_EMBED_SIZE = 35        # Safe batch size (35 items / 25s = 84 items/min)

# Global tracking
LAST_RETRIEVAL_SCORES = {}   # {(file, chunk_id): score}
CITATION_MAP = {}            # {CITE_ID: citation_string}

# Document chunking defaults (Statutory structure-aware)
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval hyperparameters
DEFAULT_TOP_K = 4
RRF_K = 60                   # Standard Reciprocal Rank Fusion constant
BM25_K1 = 1.5
BM25_B = 0.75

# Cache file
CACHE_FILE = "gemini_embeddings_cache.pkl"

# Domain validation keywords for Stage 1 fast filter
CHILD_LAW_KEYWORDS = [
    "child", "children", "minor", "pocso", "juvenile",
    "cwc", "jjb", "adoption", "guardian", "foster",
    "child labour", "child abuse", "child sexual abuse",
    "penetrative", "carnal", "trafficking", "right to education",
    "rte", "delinquent", "conflict with law", "care and protection",
    "special court", "child marriage", "tender age", "safeguard",
    "corporal punishment", "abandoned", "surrendered child"
]

# Central Statutes Manifest (Primary Bare Acts & Model Rules)
CENTRAL_ACTS_MANIFEST = {
    "POCSOact_pt1.pdf": "Protection of Children from Sexual Offences (POCSO) Act, 2012",
    "POCSOrules.pdf": "POCSO Rules, 2020",
    "jjact2015.pdf": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "juvenile_justice_rules_2017.pdf": "Juvenile Justice Model Rules, 2017",
    "constitution_english.pdf": "Constitution of India (Child Rights Provisions)"
}


def call_llm(contents: str, system_instruction: str = None) -> str:
    """Wrapper for LLM calls with automatic retry on rate limits"""
    attempts = 0
    while attempts < 5:
        try:
            config = {}
            if system_instruction:
                config["system_instruction"] = system_instruction
            
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=contents,
                config=config if config else None
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            attempts += 1
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(25)
            else:
                time.sleep(2)
            if attempts >= 5:
                raise e
    return ""


def call_embed(texts: list[str]) -> list[list[float]]:
    """Batch embed text passages using Gemini embeddings"""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    return [e.values for e in response.embeddings]

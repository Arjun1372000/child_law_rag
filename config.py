"""Central configuration and Ollama client for Child Law RAG."""

import os
import sys
import time
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "law_data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "3"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "700"))

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Cache is model-specific so changing embeddings can never silently reuse an old index.
MODEL_CACHE_NAME = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in EMBEDDING_MODEL)
CACHE_FILE = f"{MODEL_CACHE_NAME}_embeddings_cache.pkl"
CHECKPOINT_FILE = OUTPUTS_DIR / f"{MODEL_CACHE_NAME}_embeddings_checkpoint.pkl"

REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.05"))
BATCH_EMBED_SIZE = int(os.getenv("BATCH_EMBED_SIZE", "16"))

LAST_RETRIEVAL_SCORES = {}
CITATION_MAP = {}

DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_TOP_K = 5
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

CHILD_LAW_KEYWORDS = [
    "child", "children", "minor", "pocso", "juvenile", "cwc", "jjb",
    "adoption", "guardian", "foster", "child labour", "child abuse",
    "child sexual abuse", "penetrative", "trafficking", "right to education",
    "rte", "delinquent", "conflict with law", "care and protection",
    "special court", "child marriage", "tender age", "safeguard",
    "corporal punishment", "abandoned", "surrendered child"
]

CENTRAL_ACTS_MANIFEST = {
    # Put the official bare Act PDF here. The existing POCSOact_pt1.pdf in the
    # repository is a guidance/handbook and should not be treated as the Act.
    "pocso_act_2012_official.pdf": "Protection of Children from Sexual Offences (POCSO) Act, 2012",
    "POCSOrules.pdf": "Protection of Children from Sexual Offences Rules, 2020",
    "jjact2015.pdf": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "juvenile_justice_rules_2017.pdf": "Juvenile Justice Model Rules, 2017",
    "constitution_english.pdf": "Constitution of India (Child Rights Provisions)",
}


def _post_ollama(endpoint: str, payload: dict) -> dict:
    """POST to Ollama with bounded retries and useful server errors."""
    last_error = None
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            response = requests.post(
                f"{OLLAMA_HOST}{endpoint}",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            if response.status_code >= 500:
                detail = response.text[:500].strip()
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {detail}")
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, RuntimeError, requests.RequestException) as exc:
            last_error = exc
            if attempt < OLLAMA_RETRIES:
                delay = min(8.0, 1.5 ** attempt)
                print(f"  [Ollama] attempt {attempt}/{OLLAMA_RETRIES} failed: {exc}; retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise RuntimeError(f"Ollama request failed after {OLLAMA_RETRIES} attempts: {last_error}")


def call_llm(contents: str, system_instruction: str = None) -> str:
    """Generate one local response from Ollama."""
    if system_instruction:
        prompt = f"{system_instruction}\n\n{contents}"
    else:
        prompt = contents

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
        },
    }
    data = _post_ollama("/api/generate", payload)
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty LLM response")
    return text


def call_embed(texts: List[str]) -> List[List[float]]:
    """Embed one or more texts with the configured local Ollama encoder."""
    if not texts:
        return []
    data = _post_ollama(
        "/api/embed",
        {
            "model": EMBEDDING_MODEL,
            "input": texts,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
    )
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama embedding response count mismatch: expected {len(texts)}, got {len(embeddings or [])}"
        )
    return embeddings

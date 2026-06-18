# src/finsight/config.py

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

DEFAULT_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.025

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"


RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))

RERANKER_HYBRID_WEIGHT = float(
    os.getenv("RERANKER_HYBRID_WEIGHT", "0.7")
)

RERANKER_MODEL_WEIGHT = float(
    os.getenv("RERANKER_MODEL_WEIGHT", "0.3")
)

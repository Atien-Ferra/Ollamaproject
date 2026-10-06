"""Shared settings. Override any of these with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
DB_DIR = Path(os.getenv("RAG_DB_DIR", ROOT / "chroma_db"))
COLLECTION = os.getenv("RAG_COLLECTION", "pdf_docs")
EVAL_FILE = Path(os.getenv("RAG_EVAL_FILE", ROOT / "eval_questions.json"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "embeddinggemma")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", CHAT_MODEL)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("RAG_TOP_K", 4))

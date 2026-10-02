import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Models (all served locally by Ollama) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Paths ---
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
TASKS_DB = Path(os.getenv("TASKS_DB", BASE_DIR / "data" / "tasks.db"))
MCP_SERVER_PATH = BASE_DIR / "mcp_server" / "tasks_server.py"

# --- RAG settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_REWRITES = int(os.getenv("MAX_REWRITES", "2"))

# --- Optional cloud storage for documents ---
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "") or None

"""
src/config.py
-------------
Single source of truth for every tunable setting in the project.

Why a dedicated config module?
    * Every other module (loaders, embeddings, retriever, CLI, evaluation)
      imports its defaults from here, so changing e.g. the chunk size is a
      one-line edit instead of a hunt through ten files.
    * Values come from the .env file (loaded with python-dotenv) with safe
      fall-backs, so the project runs out of the box even with an empty .env
      -- the only thing that truly needs a value is ONE API key.
    * Nothing in here performs work (no model loading, no network calls);
      it is pure data, so importing it is always cheap and side-effect free.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Resolve relative to THIS file so that scripts work no matter which folder
# the user launches them from (project root, evaluation/, an IDE, ...).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (silently ignored if the file is missing).
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"                 # the user's documents (any supported format, sub-folders ok)
RESULTS_DIR = PROJECT_ROOT / "results"           # CSV reports + pipeline outputs
EVAL_DIR = PROJECT_ROOT / "evaluation"           # evaluation scripts + dataset
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma_db")

EVAL_DATA_FILE = EVAL_DIR / "evaluation_data.json"        # questions + ground truth
RAG_OUTPUTS_FILE = RESULTS_DIR / "rag_outputs.json"       # pipeline answers/contexts
MANUAL_REPORT_FILE = RESULTS_DIR / "evaluation_report.csv"
RAGAS_REPORT_FILE = RESULTS_DIR / "ragas_results.csv"
INGEST_MANIFEST_FILE = RESULTS_DIR / "ingest_manifest.json"
CHUNKS_FILE = RESULTS_DIR / "chunks.json"                 # every chunk produced by ingestion

# File extensions the ingestion pipeline knows how to read
# (see src/document_loader.py for the extension -> loader mapping).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".json", ".md", ".html", ".htm"}


# ---------------------------------------------------------------------------
# Small helpers for reading typed values from the environment
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    """Read an integer env var; fall back to `default` if unset or invalid."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var; fall back to `default` if unset or invalid."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# Which provider to use when nothing else is specified ("groq" | "huggingface" | "ollama").
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# API keys are read here ONLY so that src/llm.py has a single place to look.
# They are never printed or logged anywhere in the project.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Default model. Groq's FREE plan (Sept 2026) includes openai/gpt-oss-120b,
# openai/gpt-oss-20b, qwen/qwen3.6-27b and qwen/qwen3.8-27b (30 req/min, 8K
# tokens/min, 200K tokens/day). The Llama 3.x models are now "Enterprise" only.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct").strip()

# Ollama runs open models on this machine (https://ollama.com). No key: the
# provider is "ready" when the local server answers. Change the URL if Ollama
# runs on another machine or port.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

# Generation defaults. Low temperature = more deterministic, which is what we
# want for RAG: the model should stick to the retrieved evidence, not improvise.
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)

# The "judge" model used by RAGAS and by the manual LLM-based metrics.
RAGAS_JUDGE_PROVIDER = os.getenv("RAGAS_JUDGE_PROVIDER", LLM_PROVIDER).strip().lower()
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", GROQ_MODEL).strip()

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
# Registry of local sentence-transformers models selectable with --embedding-model.
# "dims" is recorded so the vector store can refuse to mix incompatible
# vectors (a 384-dim query can never be compared with 768-dim documents).
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dims": 384,
        "description": "Fastest; great default for small/medium corpora.",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dims": 768,
        "description": "Higher quality general-purpose embeddings (slower).",
    },
    "BAAI/bge-small-en-v1.5": {
        "dims": 384,
        "description": "Strong retrieval model, small footprint.",
    },
    "BAAI/bge-base-en-v1.5": {
        "dims": 768,
        "description": "Stronger retrieval model, larger footprint.",
    },
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# Vector database settings
# ---------------------------------------------------------------------------
VECTOR_DB = os.getenv("VECTOR_DB", "chroma").strip().lower()
# Chroma collection names must be 3-63 chars, alphanumeric + "_-". The actual
# name is derived from the embedding model (see vector_database.py) so that
# each embedding model gets its own, dimension-consistent collection.
COLLECTION_PREFIX = "rag"

# ---------------------------------------------------------------------------
# Chunking + retrieval defaults
# ---------------------------------------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)        # characters per chunk
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)  # characters shared by neighbours
TOP_K = _env_int("TOP_K", 5)                    # chunks handed to the LLM

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF
# paper (Cormack et al., 2009); larger values flatten the rank differences.
RRF_K = _env_int("RRF_K", 60)

# Default query-rewriting strategy (see src/query_rewriter.py for the list).
QUERY_REWRITE_STRATEGY = os.getenv("QUERY_REWRITE_STRATEGY", "manual_rrf").strip().lower()
# How many paraphrases the LLM multi-query rewriter asks for.
LLM_REWRITE_COUNT = _env_int("LLM_REWRITE_COUNT", 3)

# ---------------------------------------------------------------------------
# Domain vocabulary for the MANUAL (rule-based) query rewriter
# ---------------------------------------------------------------------------
# The first phrase of every list is the topic's "canonical" keyword that gets
# prefixed to the query in the keyword-anchored rewrite. The lists are
# deliberately tuned to the sample documents in data/ (an HR / company
# policy knowledge base). Extend them when you add documents on new topics.
TOPIC_KEYWORDS = {
    "annual_leave": [
        "annual leave", "vacation", "time off", "pto", "leave balance",
        "leave policy", "carryover", "carry over", "leave request",
        "vacation days", "days off", "holiday",
    ],
    "sick_leave": [
        "sick leave", "sick", "medical certificate", "illness", "unwell",
    ],
    "parental_leave": [
        "parental leave", "maternity", "paternity", "newborn", "adoption",
    ],
    "remote_work": [
        "remote work", "work from home", "wfh", "telework", "remote",
        "home office", "hybrid", "work remotely",
    ],
    "probation": [
        "probation", "probationary", "trial period", "new hire review",
        "confirmation of employment", "pass probation",
    ],
    "expense_reimbursement": [
        "expense reimbursement", "expense", "reimbursement", "reimburse",
        "reimbursed", "claim", "receipt", "travel cost", "out of pocket",
        "per diem", "paid back",
    ],
    "working_hours": [
        "working hours", "office hours", "core hours", "overtime",
        "work week", "shift", "timings",
    ],
    "it_security": [
        "it security", "password", "mfa", "two-factor", "vpn", "laptop",
        "usb", "phishing", "security policy", "device",
    ],
    "benefits": [
        "benefits", "health insurance", "insurance", "dental", "medical cover",
        "provident fund", "gratuity", "learning budget", "training budget",
    ],
    "payroll": [
        "payroll", "salary", "pay date", "paid on", "payslip", "bonus",
        "referral bonus", "increment",
    ],
    "resignation": [
        "resignation", "notice period", "resign", "termination", "exit",
        "offboarding", "final settlement",
    ],
    "company_info": [
        "company", "office location", "offices", "headquarters", "founded",
        "services", "contact", "ceo", "mission", "clients",
    ],
    "employees": [
        "employee", "employees", "staff", "department", "manager", "reports to",
        "headcount", "team", "role", "joined",
    ],
}

# Human-readable topic names used inside the "formal restatement" rewrite.
TOPIC_DISPLAY = {
    "annual_leave": "Annual Leave",
    "sick_leave": "Sick Leave",
    "parental_leave": "Parental Leave",
    "remote_work": "Remote Work",
    "probation": "Probation",
    "expense_reimbursement": "Expense Reimbursement",
    "working_hours": "Working Hours",
    "it_security": "IT Security",
    "benefits": "Employee Benefits",
    "payroll": "Payroll",
    "resignation": "Resignation and Notice Period",
    "company_info": "Company Information",
    "employees": "Employee Directory",
}

# Informal abbreviations users commonly type. Expanding them costs nothing
# and noticeably improves retrieval for short queries such as "can i wfh?".
ABBREVIATIONS = {
    "wfh": "work from home",
    "pto": "paid time off",
    "hr": "human resources",
    "mfa": "multi-factor authentication",
    "2fa": "two-factor authentication",
    "vpn": "virtual private network",
    "fri": "friday",
    "mon": "monday",
    "hq": "headquarters",
    "yr": "year",
    "yrs": "years",
    "mgr": "manager",
    "dept": "department",
    "asap": "as soon as possible",
}

# Words that carry almost no retrieval signal. Removed in the "keyword-only"
# rewrite so the embedding focuses on the content words.
STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "did", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "his", "her", "its", "our",
    "their", "what", "how", "when", "where", "why", "who", "which", "can",
    "could", "would", "should", "will", "to", "of", "for", "in", "on", "at",
    "and", "or", "if", "about", "this", "that", "these", "those", "be",
    "have", "has", "had", "get", "got", "am", "was", "were", "not", "no",
    "so", "up", "me", "need", "want", "much", "many", "there", "please",
    "tell", "know", "any", "some", "with", "from", "by", "as", "us",
}

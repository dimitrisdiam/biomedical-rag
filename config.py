# all the settings live here so I don't hardcode paths everywhere

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

CORPUS_PATH = DATA_DIR / "sample_corpus.jsonl"
EVAL_PATH = DATA_DIR / "eval_questions.jsonl"

INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

# small model, runs fine on cpu, no api key needed to embed
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chunk size in words
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30

TOP_K = 4

# needs OPENAI_API_KEY. without it the pipeline still runs, just gives an
# extractive answer instead of a generated one
LLM_MODEL = "gpt-4o-mini"

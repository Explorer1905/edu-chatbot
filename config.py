import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Dataset ----------
HF_DATASET = "KadamParth/Ncert_dataset"
SUBJECTS = ["Science", "Physics", "Chemistry", "Biology"]
GRADES = [6, 7, 8, 9, 10, 11, 12]

# ---------- Storage ----------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ncert_physics"
COLLECTION_NAME = "ncert_science"
CORPUS_CSV = "science_corpus.csv"
QA_CSV = "science_qa.csv"

# ---------- Retrieval ----------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5

# ---------- Groq ----------
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
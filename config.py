"""
config.py — one place for settings, so every step uses the same LLM and embeddings.

CONCEPT: Separating configuration from logic. Swap providers by editing .env,
not code.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads key=value pairs from the .env file into environment variables

# ---- Settings (override any of these in .env) --------------------------------
# Format is "provider:model". Examples:
#   groq:llama-3.3-70b-versatile   (free API key at console.groq.com)
#   ollama:llama3.1                (fully local, free, needs Ollama installed)
#   openai:gpt-4o-mini             (paid)
LLM_MODEL = os.getenv("LLM_MODEL", "groq:llama-3.3-70b-versatile")

# Embedding model runs locally on your machine (free, no API key).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

KB_DIR = os.getenv("KB_DIR", "knowledge_base")        # where your IT docs live
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")     # where the vector DB is saved
COLLECTION = "it_kb"
TICKETS_FILE = os.getenv("TICKETS_FILE", "tickets.json")

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80     # characters shared between neighbouring chunks
TOP_K = 4              # how many chunks to retrieve per question


def get_llm(temperature: float = 0):
    """Return a chat model. init_chat_model picks the right class from the
    'provider:model' string, so the rest of the code never changes."""
    from langchain.chat_models import init_chat_model
    return init_chat_model(LLM_MODEL, temperature=temperature)


def get_embeddings():
    """Return the embedding model that turns text into vectors."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    """Open the Chroma vector database that step2_ingest.py created."""
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

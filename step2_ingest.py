"""
STEP 2 — Build the knowledge base (the "R" in RAG, part 1: indexing)
Run:  python step2_ingest.py      (re-run whenever you add/edit docs)

You will learn:
  * Document loaders  -> read files into LangChain Document objects
  * Text splitters    -> cut long docs into chunks (why: precise retrieval, fits context)
  * Metadata          -> keep "source" so answers can cite where they came from
  * Embeddings        -> turn text into vectors (numbers) that capture meaning
  * Vector stores     -> save vectors so we can search by meaning (Chroma)
"""
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import KB_DIR, CHROMA_DIR, COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, get_embeddings


def load_documents() -> list[Document]:
    """Read every .md file in the knowledge base folder."""
    docs = []
    for path in sorted(Path(KB_DIR).glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                             metadata={"source": path.name}))
    print(f"Loaded {len(docs)} documents from {KB_DIR}/")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Two-stage split:
    1. By markdown headers, so each chunk stays on ONE topic and remembers its heading.
    2. By size, in case a section is still too long."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata["source"] = doc.metadata["source"]   # keep the file name
            chunks.extend(size_splitter.split_documents([section]))
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list[Document]):
    """Embed every chunk and store it in Chroma on disk."""
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)           # start fresh so there are no duplicates
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"Saved vector store to {CHROMA_DIR}/")
    return store


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    # Peek at one chunk so you can SEE what the splitter produced
    sample = chunks[0]
    print("\nExample chunk metadata:", sample.metadata)
    print("Example chunk text:\n", sample.page_content[:300], "...\n")

    # Peek at an embedding: just a long list of floats
    vector = get_embeddings().embed_query("How do I reset my password?")
    print(f"An embedding has {len(vector)} numbers, e.g. {[round(v, 3) for v in vector[:5]]}\n")

    build_vectorstore(chunks)

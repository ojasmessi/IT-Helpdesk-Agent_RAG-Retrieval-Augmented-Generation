"""
STEP 3 — Search the knowledge base (the "R" in RAG, part 2: retrieval)
Run:  python step3_retrieval.py

You will learn:
  * Similarity search: find chunks whose MEANING is close to the question,
    even when the words differ ("locked out" vs "too many failed sign-ins")
  * Similarity scores and why you might filter weak matches
  * Retrievers: the standard LangChain interface used inside chains/agents
"""
from config import get_vectorstore, TOP_K

store = get_vectorstore()

questions = [
    "I typed my password wrong too many times and now I can't log in",
    "GlobalProtect says authentication failed",
    "I think I clicked a dodgy link in an email",
    "What's the capital of France?",          # off-topic: notice the weak scores
    "my phone mail app stopped working",
]

for q in questions:
    print(f"\nQ: {q}")
    # Chroma returns a DISTANCE: lower = more similar.
    for doc, distance in store.similarity_search_with_score(q, k=2):
        print(f"   distance={distance:.3f}  [{doc.metadata['source']} > {doc.metadata.get('section', '')}]")

# A retriever wraps the store with a simple .invoke(question) -> list[Document]
retriever = store.as_retriever(search_kwargs={"k": TOP_K})
docs = retriever.invoke("printer not printing")
print("\nRetriever returned:", [d.metadata["source"] for d in docs])

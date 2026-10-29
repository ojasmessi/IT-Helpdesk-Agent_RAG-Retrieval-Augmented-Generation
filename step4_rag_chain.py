"""
STEP 4 — Your first RAG chain (Retrieve -> Augment -> Generate)
Run:  python step4_rag_chain.py

You will learn:
  * Augmenting a prompt with retrieved context
  * Grounding: tell the model to answer ONLY from the context (reduces hallucination)
  * Citing sources
  * RunnablePassthrough / parallel inputs in LCEL
  * The limits of a straight chain: no memory, no decisions, no "I don't know -> raise a ticket"
    -> that's why Step 5/6 move to LangGraph.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm, get_vectorstore, TOP_K

retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs) -> str:
    """Turn a list of Documents into one text block, labelled with the source."""
    return "\n\n".join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Contoso's IT help desk assistant. Answer the user's question using ONLY the "
     "context below. Give clear numbered steps where useful. If the answer is not in the "
     "context, say you don't know. End with 'Sources:' and the file names you used.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# The dict runs both branches on the same input string:
#   context  <- retriever finds docs, then format_docs turns them into text
#   question <- the original question passed straight through
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | get_llm()
    | StrOutputParser()
)

if __name__ == "__main__":
    for q in ["Outlook keeps asking for my password", "How do I install Photoshop?", "How do I fix the coffee machine?"]:
        print(f"\nQ: {q}\nA: {rag_chain.invoke(q)}")

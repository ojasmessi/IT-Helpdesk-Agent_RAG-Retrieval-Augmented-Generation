"""
STEP 1 — Talk to an LLM through LangChain
Run:  python step1_llm_basics.py

You will learn:
  * Chat models and message types (System / Human / AI)
  * Prompt templates with variables
  * LCEL: chaining pieces together with the | (pipe) operator
  * Why "temperature=0" matters for a support bot (consistent answers)
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm, LLM_MODEL

llm = get_llm()
print(f"Using model: {LLM_MODEL}\n")

# --- 1a. Raw messages ---------------------------------------------------------
# A SystemMessage sets behaviour; a HumanMessage is the user's input.
reply = llm.invoke([
    SystemMessage(content="You are a friendly IT help desk assistant. Answer in 2 sentences."),
    HumanMessage(content="My laptop is really slow. What should I try first?"),
])
print("1a. Raw reply object type:", type(reply).__name__)   # -> AIMessage
print("    Text:", reply.content, "\n")

# --- 1b. Prompt template + LCEL chain ----------------------------------------
# {variables} get filled in at run time. The | operator builds a pipeline:
# prompt -> llm -> parser (turns the AIMessage into a plain string).
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an IT help desk assistant for {company}. Be concise."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"company": "Contoso", "question": "How do I connect to office Wi-Fi?"})
print("1b. Chain answer:", answer)

# NOTE: the model does NOT know Contoso's real Wi-Fi name — it will guess or
# stay generic. That gap (no access to *your* private docs) is exactly what RAG fixes.

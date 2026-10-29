"""
agent/graph.py — the IT Help Desk agent, built with LangGraph.

Flow (an "agentic RAG" / Corrective-RAG pattern):

  START -> condense_question -> classify --+-- greeting / off_topic --> chit_chat ----------------> END
                                           +-- ticket_status --------> ticket_status --------------> END
                                           +-- new_ticket -----------> create_ticket --------------> END
                                           +-- security_incident ----> escalate --+
                                           +-- it_question ---------------------->+-> retrieve -> grade_documents
                                                                                                    |
                         relevant docs found? --yes--> generate ------------------------------------> END
                                              --no, first try--> rewrite_query -> retrieve (loop)
                                              --no, already retried--> create_ticket ---------------> END

What you learn here:
  * Messages state with the add_messages reducer (chat history that APPENDS)
  * LLM-powered routing (classification) feeding conditional edges
  * Self-checking RAG: grading retrieved docs, rewriting the query, retry loops
  * A safe fallback: when the bot doesn't know, it raises a ticket instead of making things up
  * Memory across turns with a checkpointer + thread_id
"""
import re
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm, get_vectorstore, TOP_K
from agent.tools import create_ticket, get_ticket_status

MAX_REWRITES = 1
ROUTES = ["greeting", "it_question", "security_incident", "ticket_status", "new_ticket", "off_topic"]


# ------------------------------------------------------------------ STATE ----
class AgentState(TypedDict, total=False):
    # add_messages is a REDUCER: when a node returns {"messages": [msg]},
    # LangGraph appends it to the list instead of replacing the list.
    messages: Annotated[list[AnyMessage], add_messages]
    question: str              # standalone version of the user's latest question
    route: str                 # label chosen by the classifier
    documents: list[Document]  # retrieved (then filtered) chunks
    attempts: int              # how many times we've rewritten the query
    ticket_id: str             # set if a ticket was raised this turn


# ---------------------------------------------------------------- HELPERS ----
_llm = None
_retriever = None


def llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    return _retriever


def ask(system: str, human: str, **vars) -> str:
    """Small helper: run a one-off prompt and return plain text."""
    chain = ChatPromptTemplate.from_messages([("system", system), ("human", human)]) | llm() | StrOutputParser()
    return chain.invoke(vars).strip()


def history_text(messages: list[AnyMessage], last_n: int = 6) -> str:
    """Previous turns (excluding the newest user message) as plain text."""
    lines = []
    for m in messages[:-1][-last_n:]:
        who = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


# ------------------------------------------------------------------ NODES ----
def condense_question(state: AgentState) -> dict:
    """Turn follow-ups like 'it still doesn't work' into a standalone question,
    so retrieval has something meaningful to search for."""
    latest = state["messages"][-1].content
    history = history_text(state["messages"])
    question = latest
    if history:
        question = ask(
            "Rewrite the user's latest message as a standalone IT support question using the "
            "conversation for context. If it is already standalone, return it unchanged. "
            "Return ONLY the question.",
            "Conversation:\n{history}\n\nLatest message: {latest}",
            history=history, latest=latest)
    # Reset per-turn fields (state persists between turns because of the checkpointer)
    return {"question": question, "attempts": 0, "documents": [], "ticket_id": ""}


def classify(state: AgentState) -> dict:
    raw = ask(
        "You route messages for an IT help desk. Reply with exactly ONE label:\n"
        "greeting - hello/thanks/small talk\n"
        "it_question - asking how to fix or do something IT related\n"
        "security_incident - phishing clicked, hacked, malware, lost/stolen device, suspicious MFA prompts\n"
        "ticket_status - asking about an existing ticket (e.g. INC-123ABC)\n"
        "new_ticket - explicitly asking to raise/log a ticket or speak to a human\n"
        "off_topic - not IT related",
        "{question}", question=state["question"]).lower()
    route = next((r for r in ROUTES if r in raw), "it_question")   # safe default
    return {"route": route}


def chit_chat(state: AgentState) -> dict:
    reply = ask(
        "You are Contoso's friendly IT help desk assistant. Reply briefly. If the message is not "
        "about IT, politely say you can only help with IT issues.",
        "{question}", question=state["question"])
    return {"messages": [AIMessage(content=reply)]}


def escalate(state: AgentState) -> dict:
    ticket_id = create_ticket.invoke({"summary": f"SECURITY: {state['question']}",
                                      "description": state["question"], "priority": "high"})
    msg = (f"🚨 This sounds like a security incident. I've raised **Priority 1 ticket {ticket_id}** "
           f"and the Security team has been notified. Please also call the Service Desk on extension 5000.")
    return {"messages": [AIMessage(content=msg)], "ticket_id": ticket_id}


def retrieve(state: AgentState) -> dict:
    return {"documents": retriever().invoke(state["question"])}


def grade_documents(state: AgentState) -> dict:
    """Ask the LLM whether each chunk actually helps answer the question.
    Keeps only relevant chunks — this is what makes the RAG 'self-correcting'."""
    relevant = []
    for doc in state["documents"]:
        verdict = ask(
            "You check if a document chunk is relevant to an IT support question. "
            "Answer only 'yes' or 'no'.",
            "Question: {question}\n\nChunk:\n{chunk}",
            question=state["question"], chunk=doc.page_content).lower()
        if verdict.startswith("yes"):
            relevant.append(doc)
    return {"documents": relevant}


def rewrite_query(state: AgentState) -> dict:
    better = ask(
        "Rewrite this IT support question to improve search in an internal IT knowledge base. "
        "Use common IT terms (e.g. 'account lockout', 'VPN', 'MFA'). Return ONLY the new question.",
        "{question}", question=state["question"])
    return {"question": better, "attempts": state.get("attempts", 0) + 1}


def generate(state: AgentState) -> dict:
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in state["documents"])
    answer = ask(
        "You are Contoso's IT help desk assistant. Answer using ONLY the context. Use short numbered "
        "steps when giving instructions. If the context doesn't fully answer, say what's missing and "
        "offer to raise a ticket. Do not invent links, extensions or policies.\n\n"
        "Conversation so far:\n{history}\n\nContext:\n{context}",
        "{question}",
        history=history_text(state["messages"]) or "(none)", context=context, question=state["question"])
    # Add sources in code (reliable) rather than trusting the LLM to cite them
    sources = sorted({d.metadata["source"] for d in state["documents"]})
    answer += "\n\n📚 Sources: " + ", ".join(sources)
    return {"messages": [AIMessage(content=answer)]}


def create_ticket_node(state: AgentState) -> dict:
    if state.get("ticket_id"):                # already raised (e.g. by escalate)
        return {"messages": [AIMessage(content="I couldn't find more guidance in the knowledge base — "
                                               "the Security team will contact you on your ticket.")]}
    transcript = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-10:])
    ticket_id = create_ticket.invoke({"summary": state["question"], "description": transcript})
    if state.get("route") == "new_ticket":
        msg = f"🎫 Done — I've raised ticket **{ticket_id}**. An engineer will be in touch."
    else:
        msg = (f"I couldn't find an answer to that in our IT knowledge base, so I've raised ticket "
               f"**{ticket_id}** for the Service Desk team. You can ask me for its status any time.")
    return {"messages": [AIMessage(content=msg)], "ticket_id": ticket_id}


def ticket_status(state: AgentState) -> dict:
    text = state["messages"][-1].content + " " + state["question"]
    match = re.search(r"INC-[A-Z0-9]{6}", text.upper())
    reply = (get_ticket_status.invoke({"ticket_id": match.group()}) if match
             else "Sure — what's the ticket ID? It looks like INC-1A2B3C.")
    return {"messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------- ROUTERS ----
def route_after_classify(state: AgentState) -> str:
    return {
        "greeting": "chit_chat", "off_topic": "chit_chat",
        "ticket_status": "ticket_status", "new_ticket": "create_ticket",
        "security_incident": "escalate", "it_question": "retrieve",
    }[state["route"]]


def route_after_grading(state: AgentState) -> str:
    if state["documents"]:
        return "generate"
    if state.get("attempts", 0) < MAX_REWRITES:
        return "rewrite_query"
    return "create_ticket"


# ------------------------------------------------------------------ GRAPH ----
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [("condense_question", condense_question), ("classify", classify),
                     ("chit_chat", chit_chat), ("escalate", escalate), ("retrieve", retrieve),
                     ("grade_documents", grade_documents), ("rewrite_query", rewrite_query),
                     ("generate", generate), ("create_ticket", create_ticket_node),
                     ("ticket_status", ticket_status)]:
        g.add_node(name, fn)

    g.add_edge(START, "condense_question")
    g.add_edge("condense_question", "classify")
    g.add_conditional_edges("classify", route_after_classify)
    g.add_edge("escalate", "retrieve")            # after raising P1, still give immediate guidance
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading)
    g.add_edge("rewrite_query", "retrieve")       # the retry loop
    for end_node in ["chit_chat", "generate", "create_ticket", "ticket_status"]:
        g.add_edge(end_node, END)

    # MemorySaver keeps each conversation's state in RAM, keyed by thread_id
    return g.compile(checkpointer=checkpointer or MemorySaver())


graph = build_graph()

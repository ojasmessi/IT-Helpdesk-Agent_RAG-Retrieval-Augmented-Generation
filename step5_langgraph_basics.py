"""
STEP 5 — LangGraph basics (no LLM needed!)
Run:  python step5_langgraph_basics.py

You will learn the 5 building blocks of every LangGraph app:
  1. STATE      – a typed dict shared by all nodes (the agent's "working memory")
  2. NODES      – plain Python functions: take state, return the fields they change
  3. EDGES      – fixed arrows: "after A, always go to B"
  4. CONDITIONAL EDGES – a router function picks the next node (this is the "agent" part)
  5. COMPILE + INVOKE  – turn the graph into a runnable app

Toy example: a ticket triage bot that routes by keyword.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. STATE
class TriageState(TypedDict):
    message: str
    category: str
    reply: str


# 2. NODES
def classify(state: TriageState) -> dict:
    text = state["message"].lower()
    if any(w in text for w in ["phish", "hacked", "virus", "stolen"]):
        category = "security"
    elif any(w in text for w in ["password", "vpn", "printer", "outlook", "wifi", "install", "mfa"]):
        category = "how_to"
    else:
        category = "other"
    print(f"   [classify] -> {category}")
    return {"category": category}            # only return what changed


def security(state: TriageState) -> dict:
    return {"reply": "🚨 Security issue: call extension 5000 now. Raising a P1 ticket."}


def how_to(state: TriageState) -> dict:
    return {"reply": "📘 I'll look that up in the knowledge base."}


def other(state: TriageState) -> dict:
    return {"reply": "🎫 I'll raise a general ticket for the team."}


# 4. ROUTER for the conditional edge: returns the NAME of the next node
def route(state: TriageState) -> str:
    return state["category"]


# Build the graph
builder = StateGraph(TriageState)
builder.add_node("classify", classify)
builder.add_node("security", security)
builder.add_node("how_to", how_to)
builder.add_node("other", other)

builder.add_edge(START, "classify")                          # 3. fixed edge
builder.add_conditional_edges("classify", route,             # 4. conditional edge
                              {"security": "security", "how_to": "how_to", "other": "other"})
for node in ["security", "how_to", "other"]:
    builder.add_edge(node, END)

graph = builder.compile()                                    # 5. compile

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())                  # paste into mermaid.live to see it
    for msg in ["I think I got hacked", "VPN won't connect", "Can I get a second monitor?"]:
        print(f"\nUser: {msg}")
        result = graph.invoke({"message": msg})              # 5. invoke
        print("Bot:", result["reply"])

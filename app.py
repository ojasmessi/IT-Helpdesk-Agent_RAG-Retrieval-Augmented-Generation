"""
STEP 7 — A web chat UI with Chainlit
Run:  chainlit run app.py -w        (opens http://localhost:8000 ; -w reloads on code changes)

You will learn:
  * Event handlers: @cl.on_chat_start (new chat) and @cl.on_message (each user message)
  * Per-user sessions: each browser chat gets its own LangGraph thread_id
  * Async: running the (synchronous) graph without freezing the UI
  * Showing the agent's internal steps in the UI (cl.Step) for transparency
"""
import uuid
import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph import graph

NODE_LABELS = {
    "condense_question": "Understanding your question", "classify": "Classifying request",
    "retrieve": "Searching knowledge base", "grade_documents": "Checking relevance",
    "rewrite_query": "Rephrasing search", "escalate": "Escalating to Security",
    "create_ticket": "Raising ticket", "ticket_status": "Looking up ticket",
}


@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(content="👋 Hi, I'm the Contoso IT Help Desk assistant. "
                             "Ask me about passwords, VPN, email, MFA, printers, software — "
                             "or report a security issue.").send()


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    inputs = {"messages": [HumanMessage(content=message.content)]}

    # Run graph.stream in a worker thread so the UI stays responsive,
    # then show each node that ran as a collapsible step.
    def run():
        return list(graph.stream(inputs, config, stream_mode="updates"))

    updates = await cl.make_async(run)()
    for update in updates:
        for node, changes in update.items():
            if node in NODE_LABELS:
                async with cl.Step(name=NODE_LABELS[node]) as step:
                    if changes and "route" in changes:
                        step.output = f"Category: {changes['route']}"
                    elif changes and "documents" in changes:
                        srcs = {d.metadata['source'] for d in changes['documents']}
                        step.output = f"{len(changes['documents'])} chunks: {', '.join(sorted(srcs)) or 'none'}"

    final = graph.get_state(config).values["messages"]
    # A turn may add more than one AI message (e.g. escalation notice + guidance)
    new_ai = []
    for m in reversed(final):
        if isinstance(m, HumanMessage):
            break
        new_ai.append(m.content)
    for text in reversed(new_ai):
        await cl.Message(content=text).send()

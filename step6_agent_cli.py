"""
STEP 6 — Chat with the full LangGraph agent in your terminal
Run:  python step6_agent_cli.py        (type 'quit' to exit, 'graph' to print the diagram)

You will learn:
  * Invoking a compiled graph with a config that carries a thread_id (memory per conversation)
  * stream_mode="updates": watching which node runs, in order — the best debugging tool you have
  * Reading the final state
Try:  "hi"  ->  "outlook keeps asking for my password"  ->  "still happening after that"
      "I clicked a link in a weird email"   "how do I book a meeting room?"   "status of INC-XXXXXX"
"""
import uuid
from langchain_core.messages import HumanMessage
from agent.graph import graph

config = {"configurable": {"thread_id": str(uuid.uuid4())}}   # one thread = one conversation
print("IT Help Desk Agent — type 'quit' to exit\n")

while True:
    user = input("You: ").strip()
    if user.lower() in {"quit", "exit"}:
        break
    if user.lower() == "graph":
        print(graph.get_graph().draw_mermaid())
        continue

    # stream_mode="updates" yields {node_name: what_that_node_returned} as each node finishes
    for update in graph.stream({"messages": [HumanMessage(content=user)]}, config, stream_mode="updates"):
        for node, changes in update.items():
            extra = ""
            if changes and "route" in changes:
                extra = f" -> route={changes['route']}"
            if changes and "documents" in changes:
                extra = f" -> {len(changes['documents'])} docs"
            print(f"   ⚙️  {node}{extra}")

    # Print every AI message added this turn (escalation adds two)
    new = []
    for m in reversed(graph.get_state(config).values["messages"]):
        if isinstance(m, HumanMessage):
            break
        new.append(m.content)
    for text in reversed(new):
        print(f"\nBot: {text}")
    print()

<img width="1902" height="795" alt="image" src="https://github.com/user-attachments/assets/49c0ad43-e7c9-4828-b552-81952b6fb616" />

# IT Help Desk Agent — RAG + LangChain + LangGraph (beginner build)

An IT support chatbot that answers from **your company's IT docs** (RAG), decides what to do with each
message (LangGraph), raises tickets when it can't help, escalates security incidents, and remembers
the conversation. Built in 7 small steps — run each one, read the comments, then move on.

## 0. Setup (once)

```bash
cd it-desk-agent
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   -> then add your API key
```

Pick an LLM in `.env`: **Groq** (free key, easiest), **Ollama** (fully local, needs ~8 GB RAM), or OpenAI.
Embeddings run locally for free (first run downloads a ~90 MB model).
If Groq rejects the model name, pick a current one from the models list in the Groq console.

## The 7 steps

| Step | Run | What it does | What you learn |
|---|---|---|---|
| 1 | `python step1_llm_basics.py` | Talks to the LLM | Chat models, message types, prompt templates, LCEL `\|` chains, temperature |
| 2 | `python step2_ingest.py` | Builds the knowledge base | Loaders, chunking & overlap, metadata, embeddings, vector store (Chroma) |
| 3 | `python step3_retrieval.py` | Searches by meaning | Similarity search, distance scores, retrievers |
| 4 | `python step4_rag_chain.py` | First RAG Q&A | Augmenting prompts with context, grounding, citing sources, limits of plain chains |
| 5 | `python step5_langgraph_basics.py` | Toy routing graph | State, nodes, edges, conditional edges, compile/invoke |
| 6 | `python step6_agent_cli.py` | Full agent in terminal | Agentic/corrective RAG, LLM routing, retry loops, tools, memory with thread_id, streaming node updates |
| 7 | `chainlit run app.py -w` | Web chat UI | Chainlit events, user sessions, async, showing agent steps in a UI |

## How the agent works (`agent/graph.py`)

```
START → condense_question → classify ─┬─ greeting/off_topic → chit_chat → END
                                       ├─ ticket_status → ticket_status → END
                                       ├─ new_ticket → create_ticket → END
                                       ├─ security_incident → escalate (P1 ticket) ─┐
                                       └─ it_question ──────────────────────────────┴→ retrieve → grade_documents
                                                                                        ├─ relevant → generate → END
                                                                                        ├─ none, 1st try → rewrite_query → retrieve
                                                                                        └─ none again → create_ticket → END
```

Type `graph` in the step 6 CLI to print the diagram as Mermaid (paste into https://mermaid.live).

## Things to try

- `hi` → `Outlook keeps asking for my password` → `it's still happening` (memory + question condensing)
- `I clicked a link in a weird email and typed my password` (escalation path)
- `How do I book a meeting room?` (not in KB → rewrite → ticket)
- `What's the status of INC-XXXXXX?` (use an ID it gave you)
- Add your own `.md` file to `knowledge_base/`, re-run step 2, ask about it.

## Exercises (do these to really learn it)

1. **Tune retrieval:** change `CHUNK_SIZE`/`TOP_K` in `config.py`, re-run steps 2–3, compare results.
2. **New route:** add a `hardware_request` label in `classify` and a node that asks for asset details.
3. **Structured output:** replace the string-parsing in `classify` with `llm.with_structured_output(...)` and a Pydantic model.
4. **Human-in-the-loop:** use LangGraph's `interrupt()` so the bot asks "Shall I raise a ticket?" before creating one.
5. **Tool-calling agent:** give `create_ticket` / `get_ticket_status` to the LLM with `llm.bind_tools(...)` + `ToolNode` and let the model decide when to call them.
6. **Persistence:** swap `MemorySaver` for `SqliteSaver` (`langgraph-checkpoint-sqlite`) so chats survive restarts.
7. **Evaluate:** write 15 question/expected-source pairs and measure how often retrieval returns the right file.
8. **Security (your specialty):** try prompt-injection in a KB doc ("ignore previous instructions…") and add defences.

## Project layout

```
config.py              settings + get_llm / get_embeddings / get_vectorstore
knowledge_base/        sample IT docs (fictional company "Contoso") — replace with your own
agent/graph.py         the LangGraph agent
agent/tools.py         create_ticket / get_ticket_status (JSON file; swap for ServiceNow/Jira API)
step1…step6, app.py    the learning steps
```

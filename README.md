# MyDesk Agent: a personal helpdesk for students and working professionals

Ask questions about **your own documents** (college policies, company HR rules, study notes, interview guides, IT FAQs)
and manage **tasks and deadlines**, all through one agent running on local LLMs (Ollama).

## Architecture

```
User -> FastAPI / CLI -> LangGraph agent -> Ollama (LLM + embeddings)
                              |
              +---------------+----------------+
              v                                v
        RAG branch                       Action branch
   Chroma vector store              MCP server (tasks, deadlines)
              ^
   docs/ folder or S3 bucket
```

LangGraph flow:

```
classify -+-> retrieve -> grade -+-> generate
          |                      +-> rewrite -> retrieve (max 2 retries)
          +-> act (MCP tool)
          +-> chitchat
```

| Concept | Where |
|---|---|
| LLM + embeddings | `app/config.py`, models served by Ollama |
| RAG | `app/ingest.py`, `app/rag.py`, with a grading + query-rewrite loop (corrective RAG) |
| MCP | `mcp_server/tasks_server.py` (server), `app/mcp_client.py` (client, discovers tools at runtime) |
| LangGraph | `app/graph.py` |
| Cloud | Docker image, optional S3 document source (`S3_BUCKET`), deployable to any VM or container host |
| Evaluation | `eval/run_eval.py`, 10 questions: routing accuracy, source hit rate, latency |

## Run locally

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text

python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env

python -m app.ingest           # build the vector index from data/docs
python -m app.cli              # chat in the terminal
uvicorn app.api:app --reload   # or the API: http://localhost:8000/docs
```

Put your own `.md`, `.txt` or `.pdf` files in `data/docs/` and re-run `python -m app.ingest`.

Try: "What's the penalty for late assignments?", "How many leave days do I get?",
"Add a task: submit report by 2026-10-20", "What's due this week?"

## Evaluate

```bash
set TASKS_DB=data/eval_tasks.db      # macOS/Linux: export TASKS_DB=data/eval_tasks.db
python -m eval.run_eval
```

## Docker

```bash
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec app python -m app.ingest
# API at http://localhost:8000/docs
```

## Optional: documents from S3

Set `S3_BUCKET` (and `S3_PREFIX`) in `.env`; `ingest` downloads the files first. For free local testing use
LocalStack and set `S3_ENDPOINT_URL=http://localhost:4566`.

## Design decisions

- **Chunk size 800 / overlap 100**: policies are short sections; larger chunks mixed unrelated topics. Tune in `.env`.
- **Grade + rewrite** instead of plain RAG: vague queries often retrieve the wrong chunks; one rewrite usually fixes it.
- **MCP instead of hardcoded functions**: the agent discovers tools at runtime, and the same server works with other MCP clients.
- **LangGraph instead of a plain loop**: explicit state, branching, bounded retries, and a readable trace.

## Ideas to extend

Hybrid search + reranker, human approval before write actions (LangGraph interrupts), Langfuse tracing, conversation memory, a Streamlit UI.

# MyDesk Agent: a personal helpdesk for students and working professionals

Ask questions about **your own documents** (college policies, company HR rules, study notes, interview guides, IT FAQs)
and manage **tasks and deadlines**, through one agent running entirely on local LLMs (Ollama).

## Architecture

```
User -> FastAPI / CLI -> LangGraph agent -> Ollama (LLM + embeddings)
                              |
              +---------------+----------------+
              v                                v
        RAG branch                       Action branch
   Vector index on disk             MCP server (tasks, deadlines)
              ^
   data/docs/ folder  (or an S3 bucket)
```

LangGraph flow:

```
classify -+-> retrieve -> grade -+-> generate
          |                      +-> rewrite -> retrieve   (bounded retries)
          +-> act  (calls an MCP tool)
          +-> chitchat
```

| Concept | Where |
|---|---|
| LLM + embeddings | `app/config.py`: `llama3.2:3b` + `nomic-embed-text` served by Ollama |
| RAG | `app/ingest.py`, `app/rag.py`: chunk, embed, cosine-similarity retrieval, with a grading + query-rewrite loop (corrective RAG) |
| MCP | `mcp_server/tasks_server.py` (server) and `app/mcp_client.py` (client that discovers tools at runtime) |
| LangGraph | `app/graph.py`: typed state, conditional routing, bounded retries, per-node trace |
| Cloud | Dockerfile + docker-compose, optional S3 document source (`S3_BUCKET`), runs on any VM or container host |
| Evaluation | `eval/run_eval.py`: routing accuracy, top-source hit rate, latency |

## Run locally

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text

python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env

python -m app.ingest             # build the vector index from data/docs
python -m app.cli                # chat in the terminal
uvicorn app.api:app --reload     # or the API: http://localhost:8000/docs
```

Put your own `.md`, `.txt` or `.pdf` files in `data/docs/` and re-run `python -m app.ingest`.

Try:
- "What's the penalty for late assignments?"
- "How many leave days do I get?"
- "Add a task: submit report by 2026-10-20"
- "What's due this week?"

## Evaluate

```bash
python -m eval.run_eval
```

Runs 10 questions through the agent and reports intent-routing accuracy, whether the top retrieved source
(or MCP tool) is the expected one, and average latency. Details are saved to `eval/results.json`.

### Results

| Version | Intent accuracy | Source/tool hit rate | Avg latency |
|---|---|---|---|
| v1: LLM-only routing, strict grader (`llama3.2:3b`, CPU) | 0.70 | 0.22 | 69 s |
| v2: rule + LLM routing, lenient grader, keep_alive | 1.00 | 1.00 | 36.5 s |

## Docker

```bash
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec app python -m app.ingest
# API at http://localhost:8000/docs
```

## Optional: documents from S3

Set `S3_BUCKET` (and `S3_PREFIX`) in `.env`; `ingest` downloads the files before indexing.
For free local testing use LocalStack and set `S3_ENDPOINT_URL=http://localhost:4566`.

## Design decisions and lessons learned

- **Chunk size 800 / overlap 100, top-k 3.** Policy sections are short, and larger chunks mixed unrelated topics.
  Everything is tunable in `.env`.
- **Corrective RAG (grade, then rewrite) with a bounded retry.** Vague queries sometimes retrieve the wrong chunks.
  After the retry limit the agent still answers from the best chunks instead of refusing.
- **Small-model lesson #1: a strict LLM grader rejected good context.** In v1 the 3B model marked relevant chunks as
  irrelevant, which caused pointless rewrite loops and refusals. Fix: a lenient grading prompt, fewer retries, and no
  hard refusal when documents were retrieved.
- **Small-model lesson #2: hybrid routing.** The 3B classifier sent "Add a task..." to the document branch.
  Fix: deterministic rules for obvious task requests, with the LLM (plus few-shot examples) handling the ambiguous
  ones. This is also faster, since it skips an LLM call.
- **Latency.** CPU inference is slow. `keep_alive` keeps the model loaded between requests, and each LLM call is
  kept small. A GPU or a hosted model endpoint is the next step.
- **MCP instead of hardcoded functions.** The agent discovers tools at runtime, and the same server works with any
  MCP client.
- **LangGraph instead of a plain loop.** Explicit state, branching, bounded retries, and a readable trace.
- **Vector index on disk (LangChain `InMemoryVectorStore`, persisted to JSON).** Chroma needs native `grpc` DLLs,
  which a Windows Application Control policy blocked. Retrieval is isolated in `app/rag.py`, so moving to
  Chroma or pgvector for a larger corpus is a one-file change.

## Troubleshooting

- **"Application Control policy has blocked this file" on Windows:** a native DLL in some package is blocked by your
  system policy. Avoid that package (that is why this project does not use Chroma).
- **Very slow first answer:** Ollama is loading the model; later answers are faster. Try a smaller model or set `LLM_MODEL`.
- **`No index found`:** run `python -m app.ingest` first.
- **Wrong tool or format from the model:** use a stronger model, e.g. `LLM_MODEL=qwen2.5:7b`.

## Ideas to extend

Hybrid search (BM25 + vectors) and a reranker, human approval before write actions (LangGraph interrupts),
Langfuse tracing, conversation memory, a Streamlit UI.

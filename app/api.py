"""FastAPI wrapper.  Run:  uvicorn app.api:app --reload   then open http://localhost:8000/docs"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import run_agent

app = FastAPI(title="MyDesk Agent", version="1.0")


class AskRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "MyDesk Agent is running. Open /docs to try the API."}
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    start = time.perf_counter()
    result = await run_agent(req.question)
    return {
        "answer": result.get("answer"),
        "intent": result.get("intent"),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "latency_s": round(time.perf_counter() - start, 2),
    }

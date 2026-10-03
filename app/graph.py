"""LangGraph agent.

START -> classify -+-> retrieve -> grade -+-> generate -> END
                   |                      +-> rewrite -> retrieve   (max MAX_REWRITES times)
                   +-> act (MCP tools) -> END
                   +-> chitchat -> END
"""
import json
import re
from datetime import date
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app import config, mcp_client, rag


# Rules for the obvious task requests: faster than an LLM call and more reliable on small models.
TASK_RULES = re.compile(
    r"(\b(add|create|make|set)\b.{0,25}\b(task|reminder|deadline|todo|to-do)\b"
    r"|\b(my|open|pending|all)\s+(tasks|todos|to-dos|deadlines)\b"
    r"|\b(mark|set)\b.{0,40}\b(done|completed?)\b"
    r"|\bwhat('s| is)\s+due\b|\bdue (this|next|in|today|tomorrow)\b"
    r"|\bupcoming deadlines?\b|\bremind me\b)",
    re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    docs: list
    relevant: bool
    rewrites: int
    answer: str
    sources: list[str]
    trace: list[str]


def _llm(json_mode: bool = False) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        format="json" if json_mode else None,
        keep_alive=config.KEEP_ALIVE,
    )


def _ask(system: str, user: str, json_mode: bool = False) -> str:
    resp = _llm(json_mode).invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _log(state: AgentState, msg: str) -> list[str]:
    print(f"  [trace] {msg}")
    return state.get("trace", []) + [msg]


# ---------------------------------------------------------------- nodes
def classify(state: AgentState) -> AgentState:
    # 1) cheap rules for obvious task requests, 2) LLM for everything else
    if TASK_RULES.search(state["question"]):
        intent, how = "task", "rule"
    else:
        system = (
            "Classify the user's message into exactly one intent. Reply with JSON: {\"intent\": \"...\"}.\n"
            "- \"task\": manage the user's own to-do list (add, list, complete tasks, deadlines, reminders).\n"
            "- \"question\": asks for information, rules, policies, advice or how-to "
            "(study, college, workplace, career, IT help).\n"
            "- \"chitchat\": greetings or thanks.\n"
            "Examples:\n"
            "\"Add a task to submit my report by Friday\" -> task\n"
            "\"Show my pending tasks\" -> task\n"
            "\"How should I plan my exam week?\" -> question\n"
            "\"How many leave days do I get?\" -> question\n"
            "\"My VPN is not working\" -> question\n"
            "\"Thanks!\" -> chitchat"
        )
        data = _parse_json(_ask(system, state["question"], json_mode=True))
        intent, how = data.get("intent", "question"), "llm"
        if intent not in ("task", "question", "chitchat"):
            intent = "question"
    return {
        "intent": intent,
        "search_query": state["question"],
        "rewrites": 0,
        "trace": _log(state, f"classify -> {intent} ({how})"),
    }


def retrieve(state: AgentState) -> AgentState:
    docs = rag.retrieve(state["search_query"])
    return {"docs": docs, "trace": _log(state, f"retrieve '{state['search_query']}' -> {len(docs)} chunks")}


def grade(state: AgentState) -> AgentState:
    context = "\n---\n".join(d.page_content for d in state["docs"])
    system = (
        "You grade document retrieval. Be lenient: answer true if the context mentions the topic of "
        "the question or contains facts that help answer it, even partially. Answer false only if the "
        "context is clearly about something else. Reply with JSON: {\"relevant\": true} or {\"relevant\": false}."
    )
    data = _parse_json(
        _ask(system, f"Question: {state['question']}\n\nContext:\n{context}", json_mode=True)
    )
    relevant = bool(data.get("relevant", False))
    return {"relevant": relevant, "trace": _log(state, f"grade -> relevant={relevant}")}


def rewrite(state: AgentState) -> AgentState:
    system = (
        "Rewrite the question as a short, keyword-rich search query for a document search. "
        "Reply with the query only."
    )
    new_query = _ask(system, state["question"]).strip('"')
    return {
        "search_query": new_query,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": _log(state, f"rewrite -> '{new_query}'"),
    }


def generate(state: AgentState) -> AgentState:
    docs = state.get("docs", [])
    if not docs:
        return {
            "answer": "I couldn't find anything in your documents. Add some to data/docs and run: python -m app.ingest",
            "sources": [],
            "trace": _log(state, "generate -> no documents"),
        }
    note = "" if state.get("relevant") else " (grader unsure after retries; answering from best chunks)"
    context = "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)
    system = (
        "You are MyDesk, a helpful assistant for students and working professionals. "
        "Answer ONLY using the context. Be concise. Mention the source file name in brackets "
        "like [file.md]. If the context does not contain the answer, say you could not find it in the documents."
    )
    answer = _ask(system, f"Context:\n{context}\n\nQuestion: {state['question']}")
    sources = list(dict.fromkeys(d.metadata.get("source", "?") for d in docs))  # rank order, deduped
    return {"answer": answer, "sources": sources, "trace": _log(state, f"generate -> sources {sources}{note}")}


def chitchat(state: AgentState) -> AgentState:
    answer = _ask(
        "You are MyDesk, a friendly assistant for students and professionals. Reply in one or two sentences.",
        state["question"],
    )
    return {"answer": answer, "sources": [], "trace": _log(state, "chitchat")}


async def act(state: AgentState) -> AgentState:
    """Pick an MCP tool with the LLM, call it, then summarize the result."""
    tools = await mcp_client.list_tools()
    tool_desc = "\n".join(
        f"- {t['name']}: {t['description']} | args schema: {json.dumps(t['schema'].get('properties', {}))}"
        for t in tools
    )
    system = (
        f"Today's date is {date.today().isoformat()}. Choose ONE tool for the user's request.\n"
        f"Available tools:\n{tool_desc}\n\n"
        "Reply with JSON only: {\"tool\": \"<name>\", \"arguments\": {...}}. "
        "Dates must be YYYY-MM-DD; resolve words like 'tomorrow' or 'next Friday' using today's date."
    )
    choice = _parse_json(_ask(system, state["question"], json_mode=True))
    name, args = choice.get("tool"), choice.get("arguments", {}) or {}
    if name not in {t["name"] for t in tools}:
        return {
            "answer": "I couldn't work out which task action you meant. Try: 'add task finish report by 2026-10-10'.",
            "sources": [],
            "trace": _log(state, f"act -> invalid tool choice {choice}"),
        }
    try:
        result = await mcp_client.call_tool(name, args)
    except Exception as exc:  # surface tool errors instead of crashing the graph
        result = f"Tool error: {exc}"
    trace = _log(state, f"act -> MCP tool {name}({args}) -> {result[:80]}")
    return {"answer": result, "sources": [f"mcp:{name}"], "trace": trace}


# ---------------------------------------------------------------- routing
def route_intent(state: AgentState) -> Literal["retrieve", "act", "chitchat"]:
    return {"question": "retrieve", "task": "act", "chitchat": "chitchat"}[state["intent"]]


def route_grade(state: AgentState) -> Literal["generate", "rewrite"]:
    if state["relevant"] or state.get("rewrites", 0) >= config.MAX_REWRITES:
        return "generate"
    return "rewrite"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("chitchat", chitchat)
    g.add_node("act", act)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_intent)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("chitchat", END)
    g.add_edge("act", END)
    return g.compile()


graph = build_graph()


async def run_agent(question: str) -> AgentState:
    return await graph.ainvoke({"question": question, "trace": []})
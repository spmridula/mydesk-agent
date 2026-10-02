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
    system = (
        "Classify the user's message. Reply with JSON: {\"intent\": \"...\"}.\n"
        "- \"task\": they want to add, list, complete or check tasks, deadlines or reminders.\n"
        "- \"question\": they ask for information or advice (study, policies, career, IT help).\n"
        "- \"chitchat\": greetings or small talk."
    )
    data = _parse_json(_ask(system, state["question"], json_mode=True))
    intent = data.get("intent", "question")
    if intent not in ("task", "question", "chitchat"):
        intent = "question"
    return {
        "intent": intent,
        "search_query": state["question"],
        "rewrites": 0,
        "trace": _log(state, f"classify -> {intent}"),
    }


def retrieve(state: AgentState) -> AgentState:
    docs = rag.retrieve(state["search_query"])
    return {"docs": docs, "trace": _log(state, f"retrieve '{state['search_query']}' -> {len(docs)} chunks")}


def grade(state: AgentState) -> AgentState:
    context = "\n---\n".join(d.page_content for d in state["docs"])
    system = (
        "You grade retrieval. Does the context contain information that helps answer the question? "
        "Reply with JSON: {\"relevant\": true} or {\"relevant\": false}."
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
    if not state.get("relevant") or not docs:
        return {
            "answer": "I couldn't find this in your documents. Try adding a relevant document and re-running ingest.",
            "sources": [],
            "trace": _log(state, "generate -> no relevant context"),
        }
    context = "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)
    system = (
        "You are MyDesk, a helpful assistant for students and working professionals. "
        "Answer ONLY using the context. Be concise. Mention the source file name in brackets "
        "like [file.md]. If the context is insufficient, say so."
    )
    answer = _ask(system, f"Context:\n{context}\n\nQuestion: {state['question']}")
    sources = list(dict.fromkeys(d.metadata.get("source", "?") for d in docs))  # rank order, deduped
    return {"answer": answer, "sources": sources, "trace": _log(state, f"generate -> sources {sources}")}


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

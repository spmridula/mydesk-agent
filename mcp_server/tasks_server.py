"""MCP server exposing a tiny task/deadline tracker (SQLite).

Any MCP client (our LangGraph agent, Claude Desktop, MCP Inspector...) can use these tools.
Run standalone to inspect:  npx @modelcontextprotocol/inspector python mcp_server/tasks_server.py

NOTE: stdio transport uses stdout for protocol messages, so never print() in this file.
"""
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(os.getenv("TASKS_DB", Path(__file__).resolve().parent.parent / "data" / "tasks.db"))

mcp = FastMCP("mydesk-tasks")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               title TEXT NOT NULL,
               due_date TEXT,
               category TEXT DEFAULT 'general',
               status TEXT DEFAULT 'open'
           )"""
    )
    return conn


def _fmt(row: sqlite3.Row) -> str:
    due = row["due_date"] or "no due date"
    return f"#{row['id']} [{row['status']}] {row['title']} (due: {due}, category: {row['category']})"


@mcp.tool()
def add_task(title: str, due_date: str = "", category: str = "general") -> str:
    """Add a task, assignment, or deadline. due_date must be YYYY-MM-DD (or empty)."""
    if due_date:
        try:
            date.fromisoformat(due_date)
        except ValueError:
            return f"Error: due_date '{due_date}' is not in YYYY-MM-DD format."
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (title, due_date, category) VALUES (?, ?, ?)",
            (title, due_date or None, category),
        )
        return f"Created task #{cur.lastrowid}: {title} (due: {due_date or 'no due date'})"


@mcp.tool()
def list_tasks(status: str = "open") -> str:
    """List tasks. status is 'open', 'done', or 'all'."""
    query = "SELECT * FROM tasks"
    params: tuple = ()
    if status in ("open", "done"):
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY due_date IS NULL, due_date, id"
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return "\n".join(_fmt(r) for r in rows) if rows else f"No {status} tasks."


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as done by its numeric id."""
    with _conn() as conn:
        cur = conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
        if cur.rowcount == 0:
            return f"No task with id {task_id}."
        return f"Task #{task_id} marked as done."


@mcp.tool()
def upcoming_deadlines(days: int = 7) -> str:
    """Show open tasks due within the next N days (including overdue ones)."""
    limit = (date.today() + timedelta(days=days)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' AND due_date IS NOT NULL "
            "AND due_date <= ? ORDER BY due_date",
            (limit,),
        ).fetchall()
    return "\n".join(_fmt(r) for r in rows) if rows else f"Nothing due in the next {days} days."


if __name__ == "__main__":
    mcp.run()  # stdio transport

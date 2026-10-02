"""Terminal chat.  Run:  python -m app.cli"""
import asyncio

from app.graph import run_agent


async def main() -> None:
    print("MyDesk Agent (type 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = await run_agent(question)
        print(f"\nMyDesk: {result['answer']}")
        if result.get("sources"):
            print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    asyncio.run(main())

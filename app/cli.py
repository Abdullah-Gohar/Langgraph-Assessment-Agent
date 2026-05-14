"""Interactive CLI for testing the agent without the HTTP layer.

Usage:
    python -m app.cli
"""
from __future__ import annotations

import sys
import uuid

from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph, make_checkpointer
from app.services.curriculum import get_curriculum_service


def main():
    print("Loading curriculum and embeddings (one-time cost)...")
    curr = get_curriculum_service()
    print(f"Loaded {len(curr.los)} LOs and {len(curr.chunks)} chunks.\n")

    checkpointer = make_checkpointer()
    graph = build_graph(checkpointer)

    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}")
    print("Type your message (or 'quit' to exit).\n")

    config = {"configurable": {"thread_id": session_id}}

    first_turn = True
    while True:
        try:
            msg = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break

        state_update = {
            "user_input": msg,
            "messages": [HumanMessage(content=msg)],
        }
        if first_turn:
            state_update["phase"] = "start"
            first_turn = False

        try:
            result = graph.invoke(state_update, config=config)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            continue

        print(f"\nAgent ({result.get('phase')}):\n{result.get('reply', '')}\n")
        if result.get("is_final"):
            print("--- Session complete ---")
            break


if __name__ == "__main__":
    main()

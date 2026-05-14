"""LangGraph definition for the assessment agent.

The graph is built around a router node that classifies the teacher's
intent on each turn, then conditional edges dispatch to the right
worker node. Each turn corresponds to one full pass through the graph
ending at END — the checkpointer persists state between turns and we
resume on the next API call.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.agent.state import AgentState
from app.agent.nodes import (
    router_node,
    greeting_node,
    discovery_open_node,
    discovery_topic_node,
    select_los_node,
    approve_content_node,
    refine_content_node,
    generate_node,
    off_topic_node,
)


def _route_after_router(state: AgentState) -> str:
    """Conditional edge: pick the next node based on classified intent.

    We also consider the current phase to disambiguate (a "discover_topic"
    message during content_review should be treated as refinement, etc).
    """
    intent = state.get("_intent", "discover_open")
    phase = state.get("phase", "start")

    # Hard mappings by intent first
    if intent == "greeting":
        return "greeting"
    if intent == "off_topic":
        return "off_topic"

    # Phase-aware routing for intents that depend on conversation state
    if phase in ("start", "discovery"):
        if intent == "discover_topic":
            return "discovery_topic"
        if intent == "discover_open":
            return "discovery_open"
        # If the teacher tries to select before we've offered, fall back
        return "discovery_open"

    if phase == "los_offered":
        if intent == "select_los":
            return "select_los"
        # Teacher might be refining their topic before selecting
        if intent in ("discover_topic", "discover_open"):
            return "discovery_topic" if intent == "discover_topic" else "discovery_open"
        return "select_los"

    if phase == "content_review":
        if intent == "approve_content":
            return "approve_content"
        if intent == "generate":
            return "approve_content"  # implicit approve, then ask for format
        if intent == "refine_content":
            return "refine_content"
        return "refine_content"

    if phase == "ready_to_generate":
        if intent == "generate":
            return "generate"
        if intent == "refine_content":
            return "refine_content"
        return "generate"

    if phase == "done":
        return "off_topic"  # session is closed

    return "discovery_open"


def build_graph(checkpointer):
    """Compile the StateGraph with the provided checkpointer."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router", router_node)
    builder.add_node("greeting", greeting_node)
    builder.add_node("discovery_open", discovery_open_node)
    builder.add_node("discovery_topic", discovery_topic_node)
    builder.add_node("select_los", select_los_node)
    builder.add_node("approve_content", approve_content_node)
    builder.add_node("refine_content", refine_content_node)
    builder.add_node("generate", generate_node)
    builder.add_node("off_topic", off_topic_node)

    # Entry point: always go through the router first
    builder.set_entry_point("router")

    # Conditional dispatch from router
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "greeting": "greeting",
            "discovery_open": "discovery_open",
            "discovery_topic": "discovery_topic",
            "select_los": "select_los",
            "approve_content": "approve_content",
            "refine_content": "refine_content",
            "generate": "generate",
            "off_topic": "off_topic",
        },
    )

    # All worker nodes terminate the turn — the next user message starts
    # a fresh pass from the router with the persisted state.
    for node in [
        "greeting",
        "discovery_open",
        "discovery_topic",
        "select_los",
        "approve_content",
        "refine_content",
        "generate",
        "off_topic",
    ]:
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)


def make_checkpointer() -> SqliteSaver:
    """Build a SqliteSaver that persists across process restarts."""
    settings = get_settings()
    path = Path(settings.sqlite_checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)

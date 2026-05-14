"""Conversation state held by the LangGraph agent.

This TypedDict is the single source of truth that gets checkpointed
between turns. The session_id (LangGraph's `thread_id`) maps to one
full instance of this state.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

from app.schemas import ChunkSummary, GeneratedQuestion


# Phases mirror the agent's progression through the task. The router
# uses the current phase + the teacher's message to decide where to go
# next. They're strings so they serialize cleanly in the checkpoint.
Phase = Literal[
    "start",          # nothing has happened yet
    "discovery",      # awaiting topic or selection
    "los_offered",    # we've shown candidate LOs, waiting for selection
    "content_review", # we've shown chunk summaries, waiting for approval/refinement
    "ready_to_generate",
    "done",
]


class AgentState(TypedDict, total=False):
    """Full state for one teacher session.

    `messages` uses LangGraph's add_messages reducer so each turn appends
    rather than overwrites. Everything else is overwritten by whichever
    node writes it last.
    """
    # Conversation log
    messages: Annotated[list, add_messages]

    # The teacher's latest input — convenience copy used by nodes.
    user_input: str

    # Progress through the task
    phase: Phase

    # The most recent agent reply (also pushed to `messages`).
    reply: str

    # Discovery / selection
    offered_lo_ids: list[str]           # LOs presented to the teacher
    selected_lo_ids: list[str]           # LOs the teacher has confirmed

    # Content retrieval
    chunk_summaries: list[dict]          # serialized ChunkSummary objects
    refinement_history: list[dict]       # [{"lo_id": "...", "reason": "..."}]

    # Generation
    generated_questions: list[dict]      # serialized GeneratedQuestion objects
    is_final: bool                        # True once markdown is delivered

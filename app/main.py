"""FastAPI application — exposes the agent via a single /answer endpoint.

The teacher's client sends (session_id, message) on every turn. We use
session_id as LangGraph's thread_id, which the checkpointer uses to
persist and resume state. No other endpoint is needed — the agent
itself drives the conversation through to completion.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph, make_checkpointer
from app.config import get_settings
from app.schemas import AnswerRequest, AnswerResponse
from app.services.curriculum import get_curriculum_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("assessment_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the curriculum service and graph once at startup."""
    settings = get_settings()
    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY is empty — the agent will fail on the first LLM call.")

    log.info("Loading curriculum and building embeddings...")
    curr = get_curriculum_service()
    log.info("Loaded %d LOs and %d chunks.", len(curr.los), len(curr.chunks))

    checkpointer = make_checkpointer()
    graph = build_graph(checkpointer)
    app.state.graph = graph
    app.state.checkpointer = checkpointer
    log.info("Graph compiled and ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Assessment Agent",
    description="Conversational AI agent that helps teachers create student assessments.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
async def root():
    return {"service": "assessment-agent", "status": "ok"}


@app.get("/health", tags=["health"])
async def health():
    """Simple readiness probe."""
    curr = get_curriculum_service()
    return {
        "status": "ok",
        "lo_count": len(curr.los),
        "chunk_count": len(curr.chunks),
    }


@app.post("/answer", response_model=AnswerResponse, tags=["agent"])
async def answer(req: AnswerRequest):
    """Main agent endpoint.

    Sends the teacher's message through the LangGraph agent. The graph
    routes to the right node based on the persisted state (resumed by
    session_id) and the new message. Returns the agent's reply plus the
    current phase.

    When the agent reaches the 'done' phase, is_final=True. Further
    messages with the same session_id will be politely redirected — the
    teacher should start a new session_id to begin again.
    """
    if not req.session_id.strip():
        raise HTTPException(400, "session_id must not be empty")
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")

    graph = app.state.graph
    config = {"configurable": {"thread_id": req.session_id}}

    # Prepare the state delta for this turn. The reducer on `messages`
    # will append the human message; user_input is overwritten.
    new_state = {
        "user_input": req.message,
        "messages": [HumanMessage(content=req.message)],
    }

    # Initialize phase to "start" on the very first turn for this thread.
    snapshot = graph.get_state(config)
    if not snapshot.values:
        new_state["phase"] = "start"

    try:
        final_state = graph.invoke(new_state, config=config)
    except Exception as e:
        log.exception("Graph invocation failed")
        raise HTTPException(500, f"Agent error: {e}") from e

    reply = final_state.get("reply", "")
    phase = final_state.get("phase", "start")
    is_final = bool(final_state.get("is_final", False))

    return AnswerResponse(
        session_id=req.session_id,
        reply=reply,
        phase=phase,
        is_final=is_final,
    )


@app.get("/sessions/{session_id}/state", tags=["debug"])
async def get_session_state(session_id: str):
    """Inspect persisted state for a session — useful for debugging."""
    graph = app.state.graph
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(404, "No session found with that id")
    # Drop non-JSON-serializable bits like message objects
    safe = {
        k: v for k, v in snapshot.values.items()
        if k not in ("messages",)
    }
    safe["message_count"] = len(snapshot.values.get("messages", []))
    return safe

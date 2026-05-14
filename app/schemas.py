"""Pydantic schemas for the curriculum, retrieval results, and API I/O.

These models give us validated structure for everything that flows through
the agent. Keeping them in one file makes the data shapes easy to inspect.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class LearningOutcome(BaseModel):
    """A single learning outcome from the curriculum.

    The lo_id is the dotted code at the start of the outcome text
    (e.g. "6.5.2.1.1"). We extract it during loading so we can refer to
    LOs by short ID throughout the conversation.
    """
    lo_id: str
    text: str
    domain: str
    subdomain: str
    full_domain_label: str  # original "Domain X: ... Subdomain Y: ..." string


class Chunk(BaseModel):
    """A chunk of book content."""
    chunk_id: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

class LOMatch(BaseModel):
    """An LO matched to the teacher's intent, with a similarity score."""
    lo_id: str
    score: float


class ChunkSummary(BaseModel):
    """A retrieved chunk plus an LO-conditioned summary of its content.

    During refinement we may regenerate this summary with extra context
    from the teacher's feedback, so the summary is mutable.
    """
    lo_id: str
    chunk_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    summary: str
    raw_excerpt: str  # first ~500 chars of chunk for transparency


class GeneratedQuestion(BaseModel):
    """One assessment question tied to an LO."""
    lo_id: str
    question_type: Literal["mcq", "short"]
    question: str
    options: Optional[list[str]] = None  # only for MCQ
    answer: str
    explanation: Optional[str] = None


# ---------------------------------------------------------------------------
# API I/O
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    """Request body for POST /answer."""
    session_id: str = Field(..., description="Stable per-conversation identifier.")
    message: str = Field(..., description="Teacher's message to the agent.")


class AnswerResponse(BaseModel):
    """Response body from POST /answer."""
    session_id: str
    reply: str = Field(..., description="Agent's reply, possibly markdown-formatted.")
    phase: str = Field(..., description="Current conversation phase, useful for debugging.")
    is_final: bool = Field(False, description="True when the markdown assessment is delivered and the session is closed.")

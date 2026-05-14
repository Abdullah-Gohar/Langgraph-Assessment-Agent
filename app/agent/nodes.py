"""Node implementations for the LangGraph agent.

Each node is a pure function: takes the current state and returns a
partial state update. LangGraph stitches them together using the graph
defined in graph.py.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.schemas import ChunkSummary
from app.services.curriculum import get_curriculum_service
from app.services.llm import LLMService
from app.agent.state import AgentState


# Single LLM service instance per process — cheap to share.
_llm = LLMService()


def _recent_summary(messages: list, max_chars: int = 600) -> str:
    """Build a short summary of recent turns for the router's context."""
    if not messages:
        return ""
    parts = []
    for m in messages[-6:]:  # last 3 exchanges
        role = "Teacher" if isinstance(m, HumanMessage) else "Agent"
        content = m.content if hasattr(m, "content") else str(m)
        parts.append(f"{role}: {content[:200]}")
    text = "\n".join(parts)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classify the latest message into an intent.

    The classification is stored on state and a conditional edge in
    graph.py reads it to pick the next node.
    """
    user_input = state.get("user_input", "")
    phase = state.get("phase", "start")
    history = _recent_summary(state.get("messages", []))

    intent_data = _llm.classify_intent(user_input, phase, history)
    return {"_intent": intent_data.get("intent", "discover_open")}


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------

def greeting_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    reply = _llm.greeting_reply(user_input)
    return {
        "reply": reply,
        "phase": "discovery",
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Discovery — open
# ---------------------------------------------------------------------------

def discovery_open_node(state: AgentState) -> dict:
    """Teacher is vague — show the full curriculum and invite a choice."""
    curr = get_curriculum_service()
    catalog = curr.full_catalog_markdown()
    reply = _llm.open_discovery_reply(catalog)

    offered = [lo.lo_id for lo in curr.los]
    return {
        "reply": reply,
        "phase": "los_offered",
        "offered_lo_ids": offered,
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Discovery — by topic
# ---------------------------------------------------------------------------

def discovery_topic_node(state: AgentState) -> dict:
    """Teacher described topics or student needs — match LOs semantically."""
    curr = get_curriculum_service()
    intent = state.get("user_input", "")
    matches = curr.find_matching_los(intent)

    candidate_payload = [
        (lo.lo_id, lo.full_domain_label, lo.text, score)
        for lo, score in matches
    ]
    reply = _llm.reason_over_lo_matches(intent, candidate_payload)

    offered = [lo.lo_id for lo, _ in matches]
    return {
        "reply": reply,
        "phase": "los_offered",
        "offered_lo_ids": offered,
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# LO selection
# ---------------------------------------------------------------------------

def select_los_node(state: AgentState) -> dict:
    """Parse the teacher's pick from the offered LOs and retrieve content."""
    curr = get_curriculum_service()
    offered = state.get("offered_lo_ids", [])
    selected = _llm.parse_lo_selection(state.get("user_input", ""), offered)

    if not selected:
        msg = (
            "I couldn't tell which LOs you'd like to use. Could you list the IDs "
            "(e.g. `6.5.2.1.1`, `6.5.2.1.2`) or say something like 'all of them' "
            "or 'the first three'?"
        )
        return {
            "reply": msg,
            "phase": "los_offered",
            "messages": [AIMessage(content=msg)],
        }

    # Retrieve and summarize for each selected LO
    summaries: list[ChunkSummary] = []
    for lo_id in selected:
        chunks = curr.retrieve_chunks_for_lo(lo_id)
        if not chunks:
            continue
        lo = curr.get_lo(lo_id)
        chunk_texts = [c.content for c, _ in chunks]
        summary_text = _llm.summarize_chunks_for_lo(lo.text if lo else "", chunk_texts)

        # Keep one ChunkSummary per LO, recording the top chunk for traceability.
        top_chunk, _ = chunks[0]
        summaries.append(
            ChunkSummary(
                lo_id=lo_id,
                chunk_id=top_chunk.chunk_id,
                page_start=top_chunk.page_start,
                page_end=top_chunk.page_end,
                summary=summary_text,
                raw_excerpt=top_chunk.content[:400],
            )
        )

    # Build a markdown reply that groups by subdomain
    reply = _render_content_review(curr, summaries)

    return {
        "selected_lo_ids": selected,
        "chunk_summaries": [s.model_dump() for s in summaries],
        "phase": "content_review",
        "reply": reply,
        "messages": [AIMessage(content=reply)],
    }


def _render_content_review(curr, summaries: list[ChunkSummary]) -> str:
    """Render retrieved content grouped by subdomain."""
    if not summaries:
        return "I couldn't find book content for those LOs. Could you pick others?"

    grouped = curr.group_by_subdomain([s.lo_id for s in summaries])
    summary_by_lo = {s.lo_id: s for s in summaries}

    lines = ["Here's what the textbook covers for the LOs you picked:\n"]
    for group_key, los in grouped.items():
        lines.append(f"### {group_key}")
        for lo in los:
            s = summary_by_lo.get(lo.lo_id)
            if not s:
                continue
            page_info = ""
            if s.page_start is not None:
                page_info = f" *(pages {s.page_start}–{s.page_end})*"
            lines.append(f"**`{lo.lo_id}` — {lo.text}**{page_info}")
            lines.append(f"{s.summary}")
            lines.append("")
    lines.append(
        "Does this look right? You can:\n"
        "- Say **'looks good'** to proceed to question generation\n"
        "- Reject specific LOs with a reason (e.g. *'6.5.2.1.1 is too basic, need more on real-world examples'*) and I'll find different content"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approve content
# ---------------------------------------------------------------------------

def approve_content_node(state: AgentState) -> dict:
    """Teacher approved — move to the format-selection step."""
    selected = state.get("selected_lo_ids", [])
    if not selected:
        msg = "I don't have any selected LOs yet. Let's go back and pick some."
        return {"reply": msg, "phase": "discovery", "messages": [AIMessage(content=msg)]}

    lo_list = ", ".join(f"`{x}`" for x in selected)
    msg = (
        f"Great. I'll generate questions for: {lo_list}.\n\n"
        "How would you like them? You can specify:\n"
        "- **MCQ**, **short answer**, or **mixed**\n"
        "- **1 or 2 questions per LO** (default 2)\n"
        "- Any LOs you'd like to **exclude** at this final step\n\n"
        "Or just say *'go ahead'* for a mixed assessment with 2 per LO."
    )
    return {
        "reply": msg,
        "phase": "ready_to_generate",
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Refine content
# ---------------------------------------------------------------------------

def refine_content_node(state: AgentState) -> dict:
    """Re-retrieve chunks for the rejected LOs using teacher feedback."""
    curr = get_curriculum_service()
    selected = state.get("selected_lo_ids", [])
    feedback_data = _llm.parse_refinement(state.get("user_input", ""), selected)
    rejected = feedback_data.get("rejected_los", [])

    if not rejected:
        msg = (
            "I wasn't sure which LOs you wanted to change. Could you mention the "
            "LO ID(s) and what you'd like different? For example: "
            "*'6.5.2.1.1 is too basic, I want examples of acids/bases in daily life.'*"
        )
        return {
            "reply": msg,
            "phase": "content_review",
            "messages": [AIMessage(content=msg)],
        }

    # Re-retrieve content for each rejected LO using feedback-guided query
    existing = [ChunkSummary(**s) for s in state.get("chunk_summaries", [])]
    by_lo = {s.lo_id: s for s in existing}

    new_history = list(state.get("refinement_history", []))

    for r in rejected:
        lo_id = r["lo_id"]
        reason = r["reason"]
        lo = curr.get_lo(lo_id)
        if not lo:
            continue
        hint = _llm.rewrite_query_from_feedback(lo.text, reason)
        chunks = curr.retrieve_chunks_for_lo(lo_id, extra_context=hint)
        if not chunks:
            continue
        top_chunk, _ = chunks[0]
        new_summary_text = _llm.summarize_chunks_for_lo(
            lo.text, [c.content for c, _ in chunks]
        )
        by_lo[lo_id] = ChunkSummary(
            lo_id=lo_id,
            chunk_id=top_chunk.chunk_id,
            page_start=top_chunk.page_start,
            page_end=top_chunk.page_end,
            summary=new_summary_text,
            raw_excerpt=top_chunk.content[:400],
        )
        new_history.append({"lo_id": lo_id, "reason": reason, "hint": hint})

    refreshed = list(by_lo.values())
    reply = _render_content_review(curr, refreshed)
    reply = (
        "I pulled different content for the LOs you flagged. Here's the updated picture:\n\n"
        + reply
    )

    return {
        "chunk_summaries": [s.model_dump() for s in refreshed],
        "refinement_history": new_history,
        "phase": "content_review",
        "reply": reply,
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate_node(state: AgentState) -> dict:
    """Produce the final assessment as markdown and end the session."""
    curr = get_curriculum_service()
    selected = state.get("selected_lo_ids", [])
    summaries = [ChunkSummary(**s) for s in state.get("chunk_summaries", [])]
    summary_by_lo = {s.lo_id: s for s in summaries}

    req = _llm.parse_generation_request(state.get("user_input", ""), selected)
    qtype = req["question_type"]
    excluded = set(req["excluded_lo_ids"])
    per_lo = req["per_lo_count"]

    final_los = [lo_id for lo_id in selected if lo_id not in excluded]

    all_questions = []
    for lo_id in final_los:
        lo = curr.get_lo(lo_id)
        s = summary_by_lo.get(lo_id)
        if not lo or not s:
            continue
        # Use the raw excerpt + summary as grounding for question generation
        source = f"Summary: {s.summary}\n\nExcerpt: {s.raw_excerpt}"
        qs = _llm.generate_questions_for_lo(
            lo_id=lo_id,
            lo_text=lo.text,
            source_content=source,
            question_type=qtype,
            count=per_lo,
        )
        all_questions.extend(qs)

    markdown = _render_final_assessment(curr, all_questions, excluded)

    return {
        "generated_questions": [q.model_dump() for q in all_questions],
        "phase": "done",
        "is_final": True,
        "reply": markdown,
        "messages": [AIMessage(content=markdown)],
    }


def _render_final_assessment(curr, questions, excluded: set) -> str:
    """Format the final assessment in clean markdown, grouped by LO."""
    if not questions:
        return (
            "I couldn't generate questions — there may not be enough source content. "
            "Try selecting different LOs."
        )

    lines = ["# Assessment\n"]
    grouped = curr.group_by_subdomain([q.lo_id for q in questions])
    qs_by_lo: dict[str, list] = {}
    for q in questions:
        qs_by_lo.setdefault(q.lo_id, []).append(q)

    q_num = 1
    for group_key, los in grouped.items():
        lines.append(f"## {group_key}\n")
        for lo in los:
            qs = qs_by_lo.get(lo.lo_id, [])
            if not qs:
                continue
            lines.append(f"### `{lo.lo_id}` — {lo.text}\n")
            for q in qs:
                lines.append(f"**Q{q_num}.** {q.question}")
                if q.question_type == "mcq" and q.options:
                    for opt in q.options:
                        lines.append(f"- {opt}")
                lines.append("")
                lines.append(f"<details><summary>Answer</summary>\n\n{q.answer}")
                if q.explanation:
                    lines.append(f"\n*{q.explanation}*")
                lines.append("\n</details>\n")
                q_num += 1

    if excluded:
        ex_list = ", ".join(f"`{x}`" for x in sorted(excluded))
        lines.append(f"\n*Excluded at teacher's request: {ex_list}*")

    lines.append("\n---\n*Assessment generated. This session is now closed.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Off topic
# ---------------------------------------------------------------------------

def off_topic_node(state: AgentState) -> dict:
    msg = (
        "I'm here to help you build a student assessment. Want to tell me which "
        "topics or learning outcomes you'd like to cover?"
    )
    return {
        "reply": msg,
        "messages": [AIMessage(content=msg)],
    }

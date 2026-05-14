"""LLM service — wraps OpenAI chat completions with task-specific prompts.

Each node in the LangGraph agent calls one of these methods. Keeping all
prompts here makes them easy to iterate on without touching the graph
logic.
"""
from __future__ import annotations

import json
from typing import Literal

from openai import OpenAI

from app.config import get_settings
from app.schemas import GeneratedQuestion


_SYSTEM_AGENT = """You are an assessment-design assistant for school teachers.
You help them create student assessments grounded in a specific curriculum
and textbook. Be warm, concise, and concrete. Speak like a helpful colleague,
not a chatbot. Use markdown for structured replies. Never invent learning
outcomes or book content — only use what is given to you."""


class LLMService:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, client: OpenAI | None = None):
        self.settings = get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_chat_model

    def _chat(self, messages: list[dict], temperature: float = 0.3, response_format: dict | None = None) -> str:
        """Send a chat request and return the text reply."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Intent classification + routing
    # ------------------------------------------------------------------

    def classify_intent(self, message: str, phase: str, history_summary: str) -> dict:
        """Classify the teacher's latest message into an intent.

        Returns a dict like {"intent": "...", "rationale": "..."}.
        The router uses intent to pick the next graph node.

        Possible intents:
          - "greeting"            small talk / hello
          - "discover_topic"      teacher names topics or describes what they want
          - "discover_open"       teacher wants help but is vague
          - "select_los"          teacher is choosing from previously shown LOs
          - "approve_content"     teacher accepts the retrieved content
          - "refine_content"      teacher rejects/refines with feedback
          - "generate"            teacher asks to produce the assessment
          - "off_topic"           anything unrelated
        """
        prompt = f"""Classify the teacher's message into ONE of these intents based on the current conversation phase.

Current phase: {phase}
Recent context summary: {history_summary or "(none, this is the start)"}

Intents:
- "greeting": hello, hi, thanks, casual chat
- "discover_topic": names specific topics/domains, or describes student weaknesses, or describes a past assessment that didn't go well
- "discover_open": asks for help but is vague (e.g. "I want to make an assessment")
- "select_los": picking from LOs already presented (lists IDs like 6.5.2.1.1, or says "all of them", "the first three", etc.)
- "approve_content": says the retrieved content/summaries look good, ready to proceed
- "refine_content": rejects specific LOs or asks for different content with a reason
- "generate": asks to actually create the questions, may specify MCQ vs short answer
- "off_topic": unrelated to assessment creation

Teacher's message: "{message}"

Respond with JSON only: {{"intent": "...", "rationale": "..."}}"""

        text = self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"intent": "discover_open", "rationale": "fallback — JSON parse failed"}

    # ------------------------------------------------------------------
    # Greeting reply
    # ------------------------------------------------------------------

    def greeting_reply(self, message: str) -> str:
        prompt = f"""The teacher said: "{message}"

Respond warmly and briefly (2-3 sentences). Then invite them to either:
1. Tell you the topics they want to assess, or
2. Describe the kind of assessment they have in mind.

Don't list the whole curriculum yet."""
        return self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.5,
        )

    # ------------------------------------------------------------------
    # LO matching reasoning
    # ------------------------------------------------------------------

    def reason_over_lo_matches(
        self,
        teacher_intent: str,
        candidate_los: list[tuple[str, str, str, float]],  # (lo_id, full_domain, text, score)
    ) -> str:
        """Given top-k LO matches, write a teacher-facing summary.

        Groups LOs by subdomain and explains why each is relevant.
        """
        candidates_block = "\n".join(
            f"- `{lo_id}` (similarity {score:.2f}) [{full_domain}] — {text}"
            for lo_id, full_domain, text, score in candidate_los
        )

        prompt = f"""The teacher said: "{teacher_intent}"

Here are the learning outcomes that semantically match their request, ranked by similarity:

{candidates_block}

Write a friendly markdown reply that:
1. Briefly acknowledges what they want to assess (1 sentence).
2. Presents the matching LOs grouped under their domain/subdomain headers.
3. For each LO show the ID in backticks and the outcome text. You may briefly note why a few of them fit.
4. Drop any candidate that clearly doesn't match the intent — don't pad the list.
5. End by asking which LOs they'd like to include (they can pick by ID, by subdomain, or "all of these")."""

        return self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # Selection parsing
    # ------------------------------------------------------------------

    def parse_lo_selection(self, message: str, offered_lo_ids: list[str]) -> list[str]:
        """Turn the teacher's selection message into a list of LO IDs.

        Handles things like "6.5.2.1.1 and 6.5.2.1.3", "all of them",
        "the first three", "everything in subdomain 2.1", etc.
        """
        prompt = f"""The teacher was offered these LO IDs: {offered_lo_ids}

Their selection message: "{message}"

Return JSON with the chosen IDs from the offered list, in the order they should appear.
If the teacher said "all", return all offered IDs.
If they said "first N", return the first N.
If they referenced a subdomain (e.g. "all of 2.1"), include offered IDs starting with that prefix.

Respond with JSON only: {{"selected": ["...", "..."]}}"""

        text = self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(text)
            selected = [x for x in data.get("selected", []) if x in offered_lo_ids]
            return selected
        except json.JSONDecodeError:
            return []

    # ------------------------------------------------------------------
    # Chunk summarization per LO
    # ------------------------------------------------------------------

    def summarize_chunks_for_lo(self, lo_text: str, chunks: list[str]) -> str:
        """Produce a concise summary of book content relevant to an LO."""
        chunks_block = "\n\n---\n\n".join(chunks)
        prompt = f"""Learning outcome:
{lo_text}

Book content retrieved for this outcome:
{chunks_block}

Write a 3-5 sentence summary of what this book content covers as it relates to the learning outcome.
Focus on the concepts, examples, and depth — anything that would help a teacher decide whether this material is suitable for assessment. Be concrete, no fluff."""
        return self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )

    # ------------------------------------------------------------------
    # Refinement query rewriting
    # ------------------------------------------------------------------

    def rewrite_query_from_feedback(self, lo_text: str, feedback: str) -> str:
        """Turn teacher feedback into an extra retrieval query hint."""
        prompt = f"""Learning outcome: {lo_text}

The teacher rejected the previous content with this feedback: "{feedback}"

Write a short retrieval hint (one phrase, max 20 words) that would help find more suitable book content.
Focus on what the teacher wants — examples, depth, specific subtopics, applications, etc. Output the phrase only, no quotes."""
        return self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.3,
        ).strip()

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def generate_questions_for_lo(
        self,
        lo_id: str,
        lo_text: str,
        source_content: str,
        question_type: Literal["mcq", "short", "mixed"],
        count: int = 2,
    ) -> list[GeneratedQuestion]:
        """Generate assessment questions for a single LO from grounded content."""
        type_guide = {
            "mcq": "All questions must be multiple choice with exactly 4 options.",
            "short": "All questions must be short-answer (1-3 sentence expected response).",
            "mixed": "Mix of MCQ (4 options) and short-answer is fine.",
        }[question_type]

        prompt = f"""Generate {count} assessment question(s) for this learning outcome.

LO ID: {lo_id}
LO text: {lo_text}

Source content from the textbook:
{source_content}

Requirements:
- {type_guide}
- Questions must be answerable from the source content above.
- For MCQ: 4 options labelled A, B, C, D; exactly one correct.
- Vary difficulty across questions if you generate more than one.

Respond with JSON only:
{{
  "questions": [
    {{
      "question_type": "mcq" or "short",
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."]  // omit for short
      "answer": "...",
      "explanation": "..." // optional, 1 sentence
    }}
  ]
}}"""

        text = self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(text)
            out: list[GeneratedQuestion] = []
            for q in data.get("questions", []):
                out.append(
                    GeneratedQuestion(
                        lo_id=lo_id,
                        question_type=q["question_type"],
                        question=q["question"],
                        options=q.get("options"),
                        answer=q["answer"],
                        explanation=q.get("explanation"),
                    )
                )
            return out
        except (json.JSONDecodeError, KeyError):
            return []

    # ------------------------------------------------------------------
    # Open-ended discovery (catalog)
    # ------------------------------------------------------------------

    def open_discovery_reply(self, catalog_markdown: str) -> str:
        prompt = f"""The teacher wants help creating an assessment but hasn't said what about.
Present the available curriculum naturally and invite them to choose, or to describe their students' needs.

Curriculum:
{catalog_markdown}

Keep your reply organized as markdown. Start with a one-line greeting, then show the curriculum in a clean structure (domain → subdomain → LO with IDs in backticks). End with a question inviting their input: they can pick LOs, name topics, or describe what their students need to work on."""
        return self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # Question-format parsing
    # ------------------------------------------------------------------

    def parse_generation_request(self, message: str, available_lo_ids: list[str]) -> dict:
        """Extract format + LO exclusions from a generation request."""
        prompt = f"""The teacher is asking to generate an assessment.
Available LO IDs at this point: {available_lo_ids}

Their message: "{message}"

Extract:
- question_type: "mcq", "short", or "mixed" (default "mixed" if unclear)
- excluded_lo_ids: any LOs they want to exclude (must be from the available list)
- per_lo_count: 1 or 2 (default 2 if unclear)

Respond with JSON only: {{"question_type": "...", "excluded_lo_ids": [...], "per_lo_count": ...}}"""

        text = self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(text)
            data["excluded_lo_ids"] = [x for x in data.get("excluded_lo_ids", []) if x in available_lo_ids]
            if data.get("question_type") not in ("mcq", "short", "mixed"):
                data["question_type"] = "mixed"
            if data.get("per_lo_count") not in (1, 2):
                data["per_lo_count"] = 2
            return data
        except json.JSONDecodeError:
            return {"question_type": "mixed", "excluded_lo_ids": [], "per_lo_count": 2}

    # ------------------------------------------------------------------
    # Refinement parsing
    # ------------------------------------------------------------------

    def parse_refinement(self, message: str, current_lo_ids: list[str]) -> dict:
        """Extract which LOs need refinement and the reason for each."""
        prompt = f"""The teacher is reviewing retrieved content for these LO IDs: {current_lo_ids}

Their feedback: "{message}"

Identify:
- rejected_los: list of objects {{"lo_id": "...", "reason": "..."}} for any LOs the teacher wants different content for
- approved_remaining: true if the teacher implicitly accepts the rest

Respond with JSON only: {{"rejected_los": [{{"lo_id": "...", "reason": "..."}}], "approved_remaining": true|false}}"""

        text = self._chat(
            [{"role": "system", "content": _SYSTEM_AGENT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(text)
            # Filter to known LO IDs
            data["rejected_los"] = [
                r for r in data.get("rejected_los", [])
                if r.get("lo_id") in current_lo_ids and r.get("reason")
            ]
            return data
        except json.JSONDecodeError:
            return {"rejected_los": [], "approved_remaining": False}

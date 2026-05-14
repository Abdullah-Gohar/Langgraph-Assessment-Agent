# Assessment Generation Agent

A stateful, conversational AI agent that helps teachers create student assessments grounded in a specific curriculum and textbook. Built with **FastAPI**, **LangGraph**, and **OpenAI**.

The teacher talks to the agent in natural language. The agent reasons across the curriculum (33 learning outcomes across 3 domains) to surface relevant LOs, retrieves grounded book content for them, lets the teacher review and refine, and finally produces a markdown assessment with MCQ and/or short-answer questions.

---

## Table of Contents

- [How it Works](#how-it-works)
- [Architecture](#architecture)
- [Conversation Flow](#conversation-flow)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the API](#running-the-api)
- [Running the CLI](#running-the-cli)
- [API Reference](#api-reference)
- [Sample Conversation](#sample-conversation)
- [Design Decisions](#design-decisions)
- [Testing](#testing)
- [Extending the System](#extending-the-system)

---

## How it Works

The teacher sends messages to a single endpoint, `POST /answer`, with a `session_id` and a `message`. The agent maintains conversational state per session — meaning the teacher can pause, walk away, and come back to a half-built assessment days later and pick up exactly where they left off.

Under the hood the agent is a LangGraph state machine with a router node at the top that classifies each incoming message into one of eight intents. Conditional edges then dispatch to the appropriate worker node (greeting, topic discovery, LO selection, content review, refinement, generation, etc.). Each turn ends at `END`, the state is checkpointed to SQLite, and the next message resumes from there.

Semantic matching is done via OpenAI embeddings. At startup all 33 LOs and 105 book chunks are embedded once and cached on disk. At query time the teacher's intent is embedded and matched against the LO vectors by cosine similarity — fast and accurate for this dataset size, no vector database needed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Client (teacher's browser, Postman, CLI)                         │
│   sends { session_id, message }                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI                                                          │
│   • POST /answer    (main endpoint)                              │
│   • GET  /health    (readiness)                                  │
│   • GET  /sessions/{id}/state  (debug)                           │
└──────────────────────────────┬───────────────────────────────────┘
                               │  invoke(state, thread_id=session_id)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ LangGraph agent  (SqliteSaver checkpointer)                      │
│                                                                  │
│      ┌──────────┐                                                │
│      │  router  │  classify_intent(...)                          │
│      └─────┬────┘                                                │
│            │ conditional edge by intent + phase                  │
│   ┌────────┼────────┬──────────┬──────────┬─────────┐            │
│   ▼        ▼        ▼          ▼          ▼         ▼            │
│ greet   disco_   disco_      select_    refine_   generate       │
│         open    topic        los       content                   │
│            │                   │           │          │          │
│            └───────►  content review ◄─────┘          │          │
│                            │                          │          │
│                            └──► ready ──► generate ───┤          │
│                                                       ▼          │
│                                                  END (markdown)  │
└──────────────────────────────┬───────────────────────────────────┘
                               │  reads
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Services                                                         │
│   • CurriculumService   loads LO.xlsx + chunks.json              │
│   • EmbeddingStore      OpenAI embeddings + cosine search        │
│   • LLMService          OpenAI chat with task-specific prompts   │
└──────────────────────────────────────────────────────────────────┘
```

### Four layers

| Layer        | What it does                                                                            |
| ------------ | --------------------------------------------------------------------------------------- |
| **Client**   | Sends messages. No state on the client side — just a stable `session_id`.               |
| **API**      | FastAPI. One real endpoint (`/answer`) plus health/debug helpers.                       |
| **Agent**    | LangGraph state machine. Router + worker nodes. SQLite checkpointer.                    |
| **Services** | Data loading, semantic retrieval, OpenAI chat. All prompts live in `services/llm.py`.   |

---

## Conversation Flow

The agent moves through **phases**. The current phase plus the classified intent determines where the graph routes next.

| Phase                | What's happening                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| `start`              | First turn. Teacher greets or opens.                                                                          |
| `discovery`          | Teacher hasn't picked LOs yet. Agent waiting for topic, student-needs description, or "show me everything".   |
| `los_offered`        | Agent has shown candidate LOs. Teacher selects (by ID, by subdomain, "all", "first three", etc.).             |
| `content_review`     | Agent has shown book-chunk summaries grouped by subdomain. Teacher approves or rejects-with-reason.           |
| `ready_to_generate`  | All content approved. Teacher specifies format (MCQ/short/mixed), per-LO count, any final exclusions.         |
| `done`               | Markdown assessment delivered. Session is closed.                                                             |

### Eight intents the router can classify

`greeting`, `discover_topic`, `discover_open`, `select_los`, `approve_content`, `refine_content`, `generate`, `off_topic`.

The router uses both the intent **and** the current phase, so the same message ("good, let's continue") means different things at different points in the conversation.

### The refinement loop

When the teacher rejects content with a reason (e.g. *"6.5.2.1.1 is too basic, I need real-world examples"*), the refinement node:

1. Parses the rejection into `(lo_id, reason)` pairs
2. Asks the LLM to rewrite the reason into a short retrieval hint (*"real-world applications of physical properties of matter"*)
3. Re-runs the chunk retriever with the hint appended to the query
4. Generates a fresh summary
5. Returns the teacher to `content_review` for another look

This loop can repeat as many times as needed. Refinement history is recorded in state so you can audit what was changed and why.

---

## Project Structure

```
assessment_agent/
├── app/
│   ├── __init__.py
│   ├── config.py             # Settings loaded from .env
│   ├── schemas.py            # Pydantic models (LO, Chunk, API I/O)
│   ├── data_loader.py        # Parses LO.xlsx and chunks.json
│   ├── main.py               # FastAPI app, /answer endpoint
│   ├── cli.py                # Interactive CLI for local testing
│   ├── agent/
│   │   ├── state.py          # AgentState TypedDict (the thing checkpointed)
│   │   ├── nodes.py          # All worker node implementations
│   │   └── graph.py          # StateGraph wiring + SqliteSaver factory
│   └── services/
│       ├── curriculum.py     # Singleton aggregator + retrieval helpers
│       ├── embeddings.py     # OpenAI embeddings + cosine search + cache
│       └── llm.py            # All chat prompts in one place
├── data/
│   ├── LO.xlsx               # 33 learning outcomes across 3 domains
│   ├── chunks.json           # 105 pre-chunked book pieces
│   ├── test_lo.xlsx          # The 4 LOs whose content is actually in chunks
│   └── checkpoints.sqlite    # (generated) LangGraph state persistence
├── tests/
│   └── test_data_loader.py   # Parser tests (no API key needed)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install

```bash
git clone <repo-url>
cd assessment_agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your OPENAI_API_KEY
```

The first time you run the app it will call the OpenAI embeddings API to embed all 33 LOs and 105 chunks (a one-time cost of a few cents). Results are cached to `data/embeddings_cache.json` and reused on subsequent runs unless the source data changes.

---

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","lo_count":33,"chunk_count":105}
```

---

## Running the CLI

For testing the agent without setting up an HTTP client:

```bash
python -m app.cli
```

The CLI creates a fresh session_id, then loops on stdin. Type messages, see replies, watch the phase change. Type `quit` to exit.

---

## API Reference

### POST /answer

The one endpoint that runs the whole conversation.

**Request**

```json
{
  "session_id": "teacher-abc-123",
  "message": "I want to create an assessment on matter and chemical reactions"
}
```

**Response**

```json
{
  "session_id": "teacher-abc-123",
  "reply": "Here are the LOs that match...\n\n### Domain 2: Physical Sciences — Subdomain 2.1...",
  "phase": "los_offered",
  "is_final": false
}
```

When the assessment is delivered, `is_final` is `true` and `phase` is `"done"`. Sending more messages on that session_id will get a polite "session is closed" reply — use a fresh `session_id` to start over.

### GET /health

Liveness check. Returns LO and chunk counts so you can verify data loaded.

### GET /sessions/{session_id}/state

Debug helper. Returns the persisted state (minus the full message log) for any active session.

---

## Sample Conversation

```
Teacher: Hi!

Agent (discovery):
Hello! Happy to help you build an assessment. Want to tell me which topics
you'd like to cover, or describe what your students need to work on?

──────────────────────────────────────────────────────────────────────

Teacher: My students struggled with the last quiz on chemical reactions
and the law of conservation of mass. Let's focus there.

Agent (los_offered):
Sounds like we're targeting matter and chemistry. Here are the LOs that fit:

### Domain 2: Physical Sciences — Subdomain 2.1: Matter and its Interactions
- `6.5.2.1.2` — Understanding the chemical changes of matter ... law of conservation of mass.
- `6.5.2.1.3` — Understanding chemical reactions, the indicators of their occurrence...
- `6.5.2.1.1` — Exploring the physical properties of matter...
- `6.5.2.1.4` — Exploring the chemical properties of materials... acids and bases.

Which would you like to include? You can pick by ID, by subdomain, or say "all of these".

──────────────────────────────────────────────────────────────────────

Teacher: Take the first three.

Agent (content_review):
Here's what the textbook covers for the LOs you picked:

### Domain 2: Physical Sciences — Subdomain 2.1: Matter and its Interactions
**`6.5.2.1.1` — Exploring the physical properties of matter...** (pages 12–18)
The textbook introduces physical properties (mass, volume, density)...

**`6.5.2.1.2` — Understanding the chemical changes of matter...** (pages 22–27)
Coverage focuses on conservation of mass with balance demonstrations...

**`6.5.2.1.3` — Understanding chemical reactions...** (pages 28–34)
Discusses indicators of chemical change, reaction types, and rate factors...

Does this look right? You can:
- Say 'looks good' to proceed
- Reject specific LOs with a reason and I'll find different content

──────────────────────────────────────────────────────────────────────

Teacher: 6.5.2.1.1 looks too basic. I need more on real-world examples.

Agent (content_review):
I pulled different content for the LOs you flagged. Here's the updated picture:
[refreshed summary with applied "real-world applications" focus]

──────────────────────────────────────────────────────────────────────

Teacher: Perfect. Generate 2 MCQs per LO.

Agent (done):
# Assessment

## Domain 2: Physical Sciences — Subdomain 2.1: Matter and its Interactions

### `6.5.2.1.1` — Exploring the physical properties of matter...
**Q1.** Which property of a substance ...
- A) ...
- B) ...
- C) ...
- D) ...
<details><summary>Answer</summary>B</details>

...

---
*Assessment generated. This session is now closed.*
```

---

## Design Decisions

A few choices worth calling out:

### 1. One endpoint, not many

The task said "an API with an endpoint of /answer" and the back-and-forth conversation maintains itself through session state. We honor that — `/answer` is the only real endpoint. Health and debug helpers exist but aren't part of the agent's contract.

### 2. SqliteSaver, not in-memory checkpoints

The teacher needs to be able to come back later. Using `SqliteSaver` means state survives process restarts. For production deployment, swap in `PostgresSaver` — the rest of the code doesn't change.

### 3. Embeddings cached on disk

The 33 LOs and 105 chunks are stable. Embedding them on every startup would be wasteful. We hash the content, write embeddings to `data/embeddings_cache.json`, and only re-embed if the hash changes.

### 4. LO matching via embeddings, content refinement via LLM rewriting

For matching teacher intent → LOs, embeddings are excellent: fast, deterministic, and they catch synonyms the LLM might miss. For refinement, the LLM rewrites teacher feedback ("too basic, want real-world examples") into a retrieval hint that biases the next embedding search. This is the cleanest division of labor — embeddings for retrieval, LLM for reasoning.

### 5. Router classifies intent every turn

Rather than making the graph an explicit state machine of *who can say what when*, we use the LLM to classify intent and let the routing function combine intent with phase to decide. This is more flexible — the teacher can naturally interrupt themselves ("actually wait, let me change the topics") without us needing to anticipate every reorder.

### 6. The `phase` field is the conversation memory

It's tempting to put everything in `messages` and re-derive state by asking the LLM. That gets expensive and unreliable. Instead, every node updates `phase` deterministically based on what it just did. The router reads phase to disambiguate intents that mean different things at different points.

### 7. One ChunkSummary per LO

The chunk retriever returns top-K chunks per LO (default 4) but we keep one summary per LO in state. The summary is generated from all retrieved chunks but we record only the top chunk's ID for traceability. This keeps state small while giving the question generator enough grounded content to work from.

### 8. Markdown delivery, then session close

When the assessment is generated, the session is marked `is_final` and any further message on that session_id is gently redirected. A fresh `session_id` starts a new conversation.

---

## Testing

The parser/loader tests don't require an OpenAI key and verify the data layer is wired correctly:

```bash
pytest tests/ -v
```

For integration testing the full agent, the easiest path is the CLI (`python -m app.cli`) — talk to it like a teacher would and watch the phase transitions.

---

## Extending the System

A few obvious extensions:

- **More question types**: add true/false, fill-in-blank to `LLMService.generate_questions_for_lo`. The schema and prompt are the only things that need to change.
- **Difficulty levels**: thread a `difficulty` field through state and the generation prompt.
- **Export formats**: add `/sessions/{id}/export?format=docx|pdf|qti` endpoints that pull `generated_questions` from state and format them.
- **Multi-language support**: the prompts assume English. Localize by adding a `language` field to settings and parameterizing the system prompt.
- **Postgres for production**: replace `SqliteSaver` with `PostgresSaver` in `agent/graph.py:make_checkpointer()`. State schema doesn't change.
- **Streaming responses**: LangGraph supports streaming. Add a `/answer/stream` SSE endpoint that yields tokens as the LLM produces them.

---

## License

For evaluation purposes only.

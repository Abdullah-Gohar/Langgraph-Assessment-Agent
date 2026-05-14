# Assessment Generation Agent

A stateful, conversational AI agent that helps teachers create student assessments grounded in a specific curriculum and textbook.

Built with **FastAPI**, **LangGraph**, and **OpenAI**. Ships with a built-in chat interface so you can talk to it immediately in a browser — no Postman, no extra client setup.

---

## What it does

The teacher opens the web UI and describes what they want in plain English. The agent:

1. Searches 33 curriculum learning outcomes by semantic similarity to find the ones that match the teacher's intent
2. Presents the matching LOs and lets the teacher select which to include
3. Retrieves and summarises the relevant textbook content for each selected LO
4. Lets the teacher approve or reject content — with rejection reasons fed back into a targeted re-retrieval loop
5. Generates a formatted markdown assessment (MCQ, short-answer, or both) once content is approved

Every session is persisted to SQLite so a teacher can pause mid-conversation, close the browser, and resume exactly where they left off.

---

## Quick start

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install and run

```bash
git clone <repo-url>
cd assessment_agent

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# open .env and set your OPENAI_API_KEY
```

> **First run note:** the app embeds all 33 LOs and 105 book chunks via the OpenAI embeddings API on startup. This is a one-time cost (a few cents) — results are cached to `data/embeddings_cache.json` and reused on every subsequent run.

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in a browser. You'll see the chat interface.

---

## Web interface

Navigating to `http://localhost:8000` serves a single-page chat UI directly from the API server — no separate frontend process or build step.

- Type a message and press **Enter** (or click **Send**) to talk to the agent
- Shift+Enter adds a newline without sending
- Agent replies render as formatted markdown — headers, bullet lists, code blocks, and inline formatting all display properly
- The **phase badge** in the top-right corner shows where you are in the conversation (`discovery`, `los_offered`, `content_review`, etc.)
- When the assessment is delivered the input locks and shows a "Session complete" message — refresh the page to start a new session

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Browser                                                         │
│   GET  /          → serves chat UI (app/static/index.html)      │
│   POST /answer    → sends { session_id, message }               │
│   ← renders reply as markdown                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI                                                         │
│   • POST /answer           main agent endpoint                  │
│   • GET  /health           readiness probe                      │
│   • GET  /sessions/{id}/state   debug helper                    │
│   • GET  /static/*         serves the chat UI                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  invoke(state, thread_id=session_id)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ LangGraph agent  (SqliteSaver checkpointer)                     │
│                                                                 │
│      ┌──────────┐                                               │
│      │  router  │  classify_intent(message, phase)              │
│      └─────┬────┘                                               │
│            │ conditional edge by intent + phase                 │
│   ┌────────┼──────────┬──────────┬──────────┬──────────┐        │
│   ▼        ▼          ▼          ▼          ▼          ▼        │
│ greet  disco_open  disco_topic  select_los  refine  generate    │
│                         │           │         │        │        │
│                         └──►  content_review ─┘        │        │
│                                    │                   │        │
│                                    └──► ready ─────────┘        │
│                                                        ▼        │
│                                                  END (markdown) │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Services                                                        │
│   • CurriculumService   LO.xlsx + chunks.json                   │
│   • EmbeddingStore      OpenAI embeddings + cosine search       │
│   • LLMService          all chat prompts in one place           │
└─────────────────────────────────────────────────────────────────┘
```

### Four layers

| Layer | Responsibility |
|-------|---------------|
| **Browser** | Renders the chat UI, sends messages, renders markdown replies |
| **API** | FastAPI — one real endpoint (`/answer`) plus static file serving and debug helpers |
| **Agent** | LangGraph state machine — router + worker nodes, SQLite checkpointer |
| **Services** | Data loading, semantic retrieval, all OpenAI calls |

---

## Conversation flow

The agent moves through **phases**. The current phase combined with the router's intent classification determines which node runs next.

| Phase | What's happening |
|-------|-----------------|
| `start` | First turn — teacher greets or opens |
| `discovery` | Agent waiting for a topic, a student-need description, or "show me everything" |
| `los_offered` | Candidate LOs shown — teacher selects by ID, subdomain, "all", "first three", etc. |
| `content_review` | Book-chunk summaries shown — teacher approves or rejects with a reason |
| `ready_to_generate` | All content approved — teacher specifies format and per-LO question count |
| `done` | Assessment delivered — session closed |

The router classifies each message into one of eight intents: `greeting`, `discover_topic`, `discover_open`, `select_los`, `approve_content`, `refine_content`, `generate`, `off_topic`.

### Refinement loop

When the teacher rejects content with a reason (e.g. *"6.5.2.1.1 is too basic, I need real-world examples"*), the refinement node:

1. Parses the rejection into `(lo_id, reason)` pairs
2. Uses the LLM to rewrite the reason into a focused retrieval hint
3. Re-runs the chunk retriever with the hint biasing the embedding search
4. Generates a fresh summary and returns the teacher to `content_review`

This loop repeats as many times as needed. Refinement history is recorded in state for auditability.

---

## Project structure

```
assessment_agent/
├── app/
│   ├── main.py               # FastAPI app — /answer endpoint, static file serving
│   ├── cli.py                # Interactive CLI for terminal testing
│   ├── config.py             # Settings from .env
│   ├── schemas.py            # Pydantic models (LO, Chunk, API I/O)
│   ├── data_loader.py        # Parses LO.xlsx and chunks.json
│   ├── static/
│   │   └── index.html        # Single-page chat UI (markdown rendering via marked.js)
│   ├── agent/
│   │   ├── state.py          # AgentState TypedDict — the persisted conversation state
│   │   ├── nodes.py          # All worker node implementations
│   │   └── graph.py          # StateGraph wiring + SqliteSaver factory
│   └── services/
│       ├── curriculum.py     # Singleton aggregator + retrieval helpers
│       ├── embeddings.py     # OpenAI embeddings + cosine search + disk cache
│       └── llm.py            # All chat prompts in one place (9 methods)
├── data/
│   ├── LO.xlsx               # 33 learning outcomes across 3 domains
│   ├── chunks.json           # 105 pre-chunked book excerpts
│   └── embeddings_cache.json # (generated) cached embeddings — recreated if data changes
├── tests/
│   └── test_data_loader.py   # Parser tests — no API key needed
├── requirements.txt
├── .env.example
└── README.md
```

---

## API reference

All conversation goes through a single endpoint. The web UI and the CLI both call it — the agent's behaviour is identical either way.

### `POST /answer`

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
  "reply": "Here are the learning outcomes that match...\n\n### Domain 2...",
  "phase": "los_offered",
  "is_final": false
}
```

`reply` is always markdown. When the assessment is delivered, `is_final` is `true` and `phase` is `"done"`. Further messages on the same `session_id` receive a polite redirect — start a new `session_id` to begin again.

### `GET /health`

Liveness probe. Returns LO and chunk counts.

```bash
curl http://localhost:8000/health
# {"status":"ok","lo_count":33,"chunk_count":105}
```

### `GET /sessions/{session_id}/state`

Debug helper. Returns the persisted state (minus the full message log) for any active session.

---

## Sample conversation

```
Teacher:  Hi!

Agent:    Hello! Happy to help you build an assessment. Want to tell me which
          topics you'd like to cover, or describe what your students need to
          work on?

──────────────────────────────────────────────────────────────────────────────

Teacher:  My students struggled with the last quiz on chemical reactions and
          the law of conservation of mass. Let's focus there.

Agent:    Sounds like we're targeting matter and chemistry. Here are the LOs
          that fit:

          ### Domain 2: Physical Sciences — Subdomain 2.1: Matter and its Interactions
          - `6.5.2.1.2` — Understanding the chemical changes of matter … law of conservation of mass
          - `6.5.2.1.3` — Understanding chemical reactions, indicators of their occurrence…
          - `6.5.2.1.1` — Exploring the physical properties of matter…
          - `6.5.2.1.4` — Exploring the chemical properties of materials… acids and bases

          Which would you like to include? You can pick by ID, by subdomain,
          or say "all of these".

──────────────────────────────────────────────────────────────────────────────

Teacher:  Take the first three.

Agent:    Here's what the textbook covers for those LOs:

          **`6.5.2.1.1`** — Physical properties (mass, volume, density)…
          **`6.5.2.1.2`** — Conservation of mass with balance demonstrations…
          **`6.5.2.1.3`** — Chemical change indicators, reaction types, rate factors…

          Say 'looks good' to proceed, or reject specific LOs with a reason
          and I'll find different content.

──────────────────────────────────────────────────────────────────────────────

Teacher:  6.5.2.1.1 is too basic. I need real-world examples.

Agent:    [refreshed summary with applied "real-world applications" focus]

──────────────────────────────────────────────────────────────────────────────

Teacher:  Perfect. Generate 2 MCQs per LO.

Agent:    # Assessment

          ## Domain 2 — Subdomain 2.1: Matter and its Interactions

          ### `6.5.2.1.1` — Exploring the physical properties of matter

          **Q1.** Which property of a substance …
          - A) …
          - B) …
          - C) …
          - D) …
          <details><summary>Answer</summary>B</details>

          …

          ---
          *Assessment generated. This session is now closed.*
```

---

## CLI (alternative interface)

If you prefer the terminal over the browser:

```bash
python -m app.cli
```

Creates a fresh session, loops on stdin. Type messages, see replies, watch the phase change. Type `quit` to exit. Useful for rapid testing without a browser.

---

## Design decisions

### One endpoint, not many

`/answer` is the only real endpoint. The back-and-forth conversation maintains itself through session state — the client just sends `(session_id, message)` and the agent does the rest.

### SQLite persistence, not in-memory state

`SqliteSaver` means state survives process restarts. A teacher can close the browser, come back the next day, and resume mid-conversation. For a production deployment, swap in `PostgresSaver` — no other code changes.

### Embeddings over keyword search

LO matching uses cosine similarity on OpenAI embeddings, cached on disk after the first run. This catches synonyms and paraphrases that keyword search would miss, and it's fast enough for 33 LOs with no vector database.

### LLM-rewritten retrieval hints for refinement

When a teacher rejects content, the LLM rewrites the rejection reason into a compact retrieval hint before re-running the embedding search. This is more effective than searching on the raw rejection text.

### Phase as explicit conversation memory

Rather than re-deriving state from the full message history each turn (expensive and unreliable), each worker node updates `phase` deterministically after it finishes. The router reads phase to disambiguate intents that mean different things at different stages.

---

## Testing

Parser and loader tests run without an OpenAI key:

```bash
pytest tests/ -v
```

For end-to-end testing, use the web UI or the CLI and step through a full conversation.

---

## Extending the system

| Extension | Where to change |
|-----------|----------------|
| More question types (true/false, fill-in-blank) | `LLMService.generate_questions_for_lo` — schema + prompt only |
| Difficulty levels | Add a `difficulty` field to `AgentState` and thread it through the generation prompt |
| Export formats (docx, PDF) | Add `/sessions/{id}/export` endpoints that read `generated_questions` from state |
| Multi-language support | Add a `language` field to settings and parameterise the system prompts |
| Production database | Replace `SqliteSaver` with `PostgresSaver` in `agent/graph.py:make_checkpointer()` |
| Streaming responses | LangGraph supports token streaming — add a `/answer/stream` SSE endpoint |

---

## License

For evaluation purposes only.

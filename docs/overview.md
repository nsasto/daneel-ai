# Daneel Assistant Skeleton Overview

## What exists
- `pyproject.toml`: project metadata, FastAPI/Pydantic deps, optional LangGraph extra.
- `src/daneel/models.py`: shared TypedDict state, enums, retrieval chunk and tool planning types.
- `src/daneel/clients.py`: in-memory Memobase, RAGdoll, and Graph RAG placeholders plus HTTP clients for real services (env-configurable).
- `src/daneel/config.py`: environment-driven client settings (URLs, API keys, timeouts).
- `src/daneel/tooling.py`: tool registry plus stub tools (create task, send email, schedule meeting, trigger n8n).
- `src/daneel/nodes.py`: LangGraph node functions for interaction/intent/topic classification, ingestion writes, retrieval routing, tool planning/execution, answer generation; NodeContext bootstraps default tools.
- `src/daneel/graph_builder.py`: LangGraph wiring (with fallback shim if LangGraph is missing) for ingestion + reasoning flows, conditional branching, rerank, and finish.
- `src/daneel/service.py`: FastAPI app factory and `/assistant` endpoint returning answer, topic, retrieval intent, used chunks, and tool results.
- `tests/test_assistant.py`: async smoke test exercising the compiled graph end-to-end with in-memory clients.

## How to run locally
- Install deps: `pip install -e .[dev,graph]`
- Run tests: `pytest`
- Run API (example): `uvicorn daneel.service:app --reload`

## Client configuration
- Memobase: `MEMOBASE_URL`, `MEMOBASE_API_KEY`, `MEMOBASE_TIMEOUT` (seconds)
- RAGdoll: `RAGDOLL_URL`, `RAGDOLL_API_KEY`, `RAGDOLL_TIMEOUT`
- Graph RAG: `GRAPH_RAG_URL`, `GRAPH_RAG_API_KEY`, `GRAPH_RAG_TIMEOUT`
- If a service URL is absent, the app falls back to in-memory clients for local/dev.

## Gaps / next steps
- Replace in-memory clients with real Memobase, RAGdoll, and Graph implementations.
- Add prompts and richer logic for classification, retrieval routing, and tool planning.
- Extend ingestion (entity/relationship extraction, chunking strategy) and retrieval (rerank/scoring).
- Add more tools and integrate optional ingestion of tool outputs with real stores.
- Expand tests across nodes, error cases, and API contract.

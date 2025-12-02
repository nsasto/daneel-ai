from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .clients import GraphClient, MemobaseClient, RagdollClient
from .models import (
    DEFAULT_TOPICS,
    GraphDoc,
    GraphState,
    InteractionKind,
    MemobaseRecord,
    PlannedToolCall,
    RetrievalChunk,
    RetrievalIntent,
)
from .tooling import ToolRegistry, default_tool_registry


@dataclass
class NodeContext:
    memobase: MemobaseClient
    ragdoll: RagdollClient
    graph: GraphClient
    tools: Optional[ToolRegistry]
    vector_collection_prefix: str = "daneel"

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = default_tool_registry()


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def classify_interaction(state: GraphState) -> GraphState:
    query = state.get("query", "").lower()
    if "store only" in query:
        kind = InteractionKind.STORE_ONLY
    elif "store" in query and "summarize" in query:
        kind = InteractionKind.TRANSFORM_AND_STORE
    elif "remind" in query or "now" in query:
        kind = InteractionKind.ACT_NOW
    else:
        kind = InteractionKind.ACT_AND_STORE
    state["interaction_kind"] = kind
    return state


def normalise_input(state: GraphState) -> GraphState:
    state["query"] = state.get("query", "").strip()
    return state


def detect_ingestion_type(state: GraphState) -> GraphState:
    query = state.get("query", "").lower()
    if "task" in query or "todo" in query:
        ingestion = "task"
    elif "call" in query or "meeting" in query:
        ingestion = "transcript"
    elif "contact" in query:
        ingestion = "contact"
    else:
        ingestion = "note"
    state["ingestion_type"] = ingestion
    return state


def transform_for_storage(state: GraphState) -> GraphState:
    text = state.get("query", "")
    truncated = text[:500]
    summary = truncated if len(text) <= 500 else truncated + "..."
    metadata = {
        "summary": summary,
        "length": len(text),
        "created_at": time.time(),
        "ingestion_type": state.get("ingestion_type"),
    }
    state["metadata"] = metadata  # type: ignore[typeddict-item]
    return state


def write_memobase(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        record: MemobaseRecord = {
            "id": _short_hash(state.get("query", "") + str(time.time())),
            "user_id": state["user_id"],
            "topic": state.get("primary_topic") or "general",
            "type": state.get("ingestion_type") or "note",
            "content": state.get("query", ""),
            "metadata": state.get("metadata", {}),
            "created_at": time.time(),
            "last_seen_at": time.time(),
            "strength": 1.0,
            "expires_at": None,
        }
        await ctx.memobase.write([record])
        state["memobase_results"] = [
            {
                "content": record["content"],
                "score": record["strength"],
                "source": "memobase",
                "metadata": record["metadata"],
            }
        ]
        return state

    return _node


def write_vector_indexes(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        collection = f"{ctx.vector_collection_prefix}.{state.get('primary_topic') or 'general'}"
        doc_id = _short_hash(state.get("query", ""))
        await ctx.ragdoll.ingest(
            [{"doc_id": doc_id, "text": state.get("query", ""), "metadata": {"topic": state.get("primary_topic")}}],
            collection=collection,
            enable_graph=True,
        )
        return state

    return _node


def write_graph(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        doc_id = _short_hash(state.get("query", ""))
        await ctx.graph.update_from_entities(
            entities=[
                {"name": "note", "type": "doc", "metadata": {"doc_id": doc_id, "topic": state.get("primary_topic")}}
            ],
            relationships=[],
            doc_refs=[{"doc_id": doc_id, "topic": state.get("primary_topic") or "general"}],
        )
        return state

    return _node


def classify_topic(state: GraphState) -> GraphState:
    query = state.get("query", "").lower()
    candidates = [topic for topic in DEFAULT_TOPICS if topic in query]
    primary = candidates[0] if candidates else "general"
    state["topic_candidates"] = candidates or ["general"]
    state["primary_topic"] = primary
    state["topic_confidence"] = 0.65 if candidates else 0.35
    return state


def classify_intent(state: GraphState) -> GraphState:
    query = state.get("query", "").lower()
    if any(word in query for word in ["who", "what", "when", "where"]):
        intent = RetrievalIntent.FACT_LOOKUP
    elif "how" in query or "why" in query:
        intent = RetrievalIntent.MULTI_HOP
    elif "remember" in query or "recall" in query:
        intent = RetrievalIntent.MEMORY_PERSONAL
    else:
        intent = RetrievalIntent.MIXED
    state["retrieval_intent"] = intent
    return state


def route_retrieval(state: GraphState) -> GraphState:
    intent = state.get("retrieval_intent", RetrievalIntent.MEMORY_PERSONAL)
    state["need_memobase"] = intent in {RetrievalIntent.MEMORY_PERSONAL, RetrievalIntent.MIXED}
    state["need_vector"] = intent in {RetrievalIntent.FACT_LOOKUP, RetrievalIntent.MIXED, RetrievalIntent.MULTI_HOP}
    state["need_graph"] = intent in {RetrievalIntent.MULTI_HOP, RetrievalIntent.MIXED}
    return state


def retrieve_memobase(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        hits = await ctx.memobase.search(
            user_id=state["user_id"], query=state.get("query", ""), topic=state.get("primary_topic"), k=5
        )
        state["memobase_results"] = [
            {"content": hit["content"], "score": hit["score"], "source": "memobase", "metadata": hit["metadata"]}
            for hit in hits
        ]
        return state

    return _node


def retrieve_vector(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        collection = f"{ctx.vector_collection_prefix}.{state.get('primary_topic') or 'general'}"
        hits = await ctx.ragdoll.search(
            query=state.get("query", ""),
            collections=[collection],
            k=5,
            use_graph=False,
            use_pagerank=False,
            hybrid_mode=None,
        )
        state["vector_results"] = [
            {"content": hit["text"], "score": hit["score"], "source": "vector", "metadata": hit["metadata"]}
            for hit in hits
        ]
        return state

    return _node


def retrieve_graph_pagerank(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        graphs = [state.get("primary_topic")] if state.get("primary_topic") else []
        hits: List[GraphDoc] = await ctx.graph.query_with_pagerank(
            query=state.get("query", ""), topic_graphs=graphs, max_docs=5
        )
        state["graph_results"] = [
            {"content": hit["text"], "score": hit["score"], "source": "graph", "metadata": hit["metadata"]}
            for hit in hits
        ]
        return state

    return _node


def rerank(state: GraphState) -> GraphState:
    seen: set[str] = set()
    combined: List[RetrievalChunk] = []
    for source_key in ["memobase_results", "vector_results", "graph_results"]:
        for chunk in state.get(source_key, []):
            key = chunk["content"]
            if key in seen:
                continue
            seen.add(key)
            combined.append(chunk)
    combined.sort(key=lambda c: c.get("score", 0), reverse=True)
    state["final_results"] = combined
    return state


def tool_planner(state: GraphState) -> GraphState:
    query = state.get("query", "").lower()
    planned: List[PlannedToolCall] = []
    if "task" in query or "todo" in query:
        planned.append({"tool_name": "create_task", "arguments": {"title": state.get("query")}})
    elif "email" in query:
        planned.append({"tool_name": "send_email", "arguments": {"body": state.get("query")}})
    state["planned_tools"] = planned
    return state


def run_tools(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        results: Dict[str, Any] = {}
        for call in state.get("planned_tools", []):
            tool = ctx.tools.get(call["tool_name"])
            if not tool:
                results[call["tool_name"]] = {"error": "tool not registered"}
                continue
            results[call["tool_name"]] = await tool(**call.get("arguments", {}))
        state["tool_results"] = results
        return state

    return _node


def ingest_tool_results(ctx: NodeContext) -> Callable[[GraphState], GraphState]:
    async def _node(state: GraphState) -> GraphState:
        if not state.get("tool_results"):
            return state
        serialized = str(state["tool_results"])
        record: MemobaseRecord = {
            "id": _short_hash(serialized),
            "user_id": state["user_id"],
            "topic": state.get("primary_topic") or "general",
            "type": "tool_result",
            "content": serialized,
            "metadata": {"source": "tool_ingest"},
            "created_at": time.time(),
            "last_seen_at": time.time(),
            "strength": 1.0,
            "expires_at": None,
        }
        await ctx.memobase.write([record])
        return state

    return _node


def generate_answer(state: GraphState) -> GraphState:
    answer_parts: List[str] = []
    if state.get("planned_tools"):
        answer_parts.append("Planned tools executed.")
    if state.get("final_results"):
        answer_parts.append("Found supporting context.")
    if not answer_parts:
        answer_parts.append("Processed request.")
    state["answer"] = " ".join(answer_parts)
    return state

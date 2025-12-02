from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .clients import (
    GraphClient,
    HTTPGraphClient,
    HTTPMemobaseClient,
    HTTPRagdollClient,
    InMemoryGraphClient,
    InMemoryMemobaseClient,
    InMemoryRagdollClient,
    MemobaseClient,
    RagdollClient,
)
from .config import AppSettings, load_settings
from .graph_builder import build_assistant_graph
from .models import GraphState, RetrievalIntent
from .nodes import NodeContext
from .tooling import ToolRegistry, default_tool_registry


class AssistantRequest(BaseModel):
    user_id: str = Field(..., description="Stable user identifier")
    message: str = Field(..., description="User message")
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssistantResponse(BaseModel):
    answer: str
    topic: Optional[str]
    retrieval_intent: Optional[RetrievalIntent]
    used_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)


def create_app(
    *,
    memobase: Optional[MemobaseClient] = None,
    ragdoll: Optional[RagdollClient] = None,
    graph_client: Optional[GraphClient] = None,
    tools: Optional[ToolRegistry] = None,
    settings: Optional[AppSettings] = None,
) -> FastAPI:
    app = FastAPI(title="Daneel Assistant")

    settings = settings or load_settings()
    memobase = memobase or (
        HTTPMemobaseClient(
            base_url=settings.memobase.base_url, api_key=settings.memobase.api_key, timeout=settings.memobase.timeout
        )
        if settings.memobase.enabled
        else InMemoryMemobaseClient()
    )
    ragdoll = ragdoll or (
        HTTPRagdollClient(
            base_url=settings.ragdoll.base_url, api_key=settings.ragdoll.api_key, timeout=settings.ragdoll.timeout
        )
        if settings.ragdoll.enabled
        else InMemoryRagdollClient()
    )
    graph_client = graph_client or (
        HTTPGraphClient(
            base_url=settings.graph.base_url, api_key=settings.graph.api_key, timeout=settings.graph.timeout
        )
        if settings.graph.enabled
        else InMemoryGraphClient()
    )

    ctx = NodeContext(
        memobase=memobase,
        ragdoll=ragdoll,
        graph=graph_client,
        tools=tools or default_tool_registry(),
    )
    compiled_graph = build_assistant_graph(ctx)

    async def run_graph(initial_state: GraphState) -> GraphState:
        if hasattr(compiled_graph, "ainvoke"):
            return await compiled_graph.ainvoke(initial_state)  # type: ignore[call-arg]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compiled_graph.invoke, initial_state)

    @app.post("/assistant", response_model=AssistantResponse)
    async def assistant_endpoint(req: AssistantRequest) -> AssistantResponse:
        state: GraphState = {
            "user_id": req.user_id,
            "query": req.message,
            "history": req.history,
        }
        try:
            final_state = await run_graph(state)
            return AssistantResponse(
                answer=final_state.get("answer", "Processed request"),
                topic=final_state.get("primary_topic"),
                retrieval_intent=final_state.get("retrieval_intent"),
                used_chunks=final_state.get("final_results", []),
                tool_results=final_state.get("tool_results", {}),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()

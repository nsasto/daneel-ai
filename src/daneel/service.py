from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .clients import GraphClient, MemobaseClient, RagdollClient
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
) -> FastAPI:
    app = FastAPI(title="Daneel Assistant")

    ctx = NodeContext(
        memobase=memobase or MemobaseClient(),
        ragdoll=ragdoll or RagdollClient(),
        graph=graph_client or GraphClient(),
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
        final_state = await run_graph(state)
        return AssistantResponse(
            answer=final_state.get("answer", "Processed request"),
            topic=final_state.get("primary_topic"),
            retrieval_intent=final_state.get("retrieval_intent"),
            used_chunks=final_state.get("final_results", []),
            tool_results=final_state.get("tool_results", {}),
        )

    return app


app = create_app()

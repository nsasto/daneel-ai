from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from .nodes import (
    NodeContext,
    classify_intent,
    classify_interaction,
    classify_topic,
    detect_ingestion_type,
    generate_answer,
    ingest_tool_results,
    normalise_input,
    rerank,
    retrieve_graph_pagerank,
    retrieve_memobase,
    retrieve_vector,
    route_retrieval,
    run_tools,
    tool_planner,
    transform_for_storage,
    write_graph,
    write_memobase,
    write_vector_indexes,
)
from .models import GraphState, InteractionKind

try:  # pragma: no cover - executed only when LangGraph is installed
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - lightweight fallback

    class StateGraph:
        def __init__(self, state_type: Any) -> None:
            self.state_type = state_type
            self.nodes: Dict[str, Callable[[GraphState], Awaitable[GraphState]] | Callable[[GraphState], GraphState]] = (
                {}
            )
            self.edges: Dict[str, list[str]] = {}
            self.conditional_edges: Dict[str, tuple[Callable[[GraphState], str], Dict[str, Optional[str]]]] = {}
            self.entrypoint: Optional[str] = None
            self.finish_node: Optional[str] = None

        def add_node(self, name: str, func: Callable[[GraphState], Awaitable[GraphState]] | Callable[[GraphState], GraphState]) -> None:
            self.nodes[name] = func

        def set_entry_point(self, name: str) -> None:
            self.entrypoint = name

        def set_finish_point(self, name: str) -> None:
            self.finish_node = name

        def add_edge(self, source: str, target: str) -> None:
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(
            self,
            source: str,
            condition: Callable[[GraphState], str],
            edges: Dict[str, Optional[str]],
        ) -> None:
            self.conditional_edges[source] = (condition, edges)

        def compile(self) -> "CompiledGraph":
            return CompiledGraph(self)

    class CompiledGraph:
        def __init__(self, graph: StateGraph) -> None:
            self.graph = graph

        def invoke(self, state: GraphState) -> GraphState:
            return asyncio.run(self.ainvoke(state))

        async def ainvoke(self, state: GraphState) -> GraphState:
            current = self.graph.entrypoint
            while current:
                node_func = self.graph.nodes[current]
                if asyncio.iscoroutinefunction(node_func):
                    state = await node_func(state)
                else:
                    state = node_func(state)
                if current in self.graph.conditional_edges:
                    condition, mapping = self.graph.conditional_edges[current]
                    label = condition(state)
                    current = mapping.get(label)
                    continue
                next_nodes = self.graph.edges.get(current, [])
                current = next_nodes[0] if next_nodes else None
                if current == getattr(self.graph, "finish_node", None):
                    break
            return state

    END = "__END__"


NodeFn = Callable[[GraphState], Awaitable[GraphState]] | Callable[[GraphState], GraphState]


def _guarded(node: NodeFn, flag_key: str) -> NodeFn:
    async def _wrapped(state: GraphState) -> GraphState:
        if not state.get(flag_key):
            return state
        if asyncio.iscoroutinefunction(node):
            return await node(state)  # type: ignore[arg-type]
        return node(state)  # type: ignore[arg-type]

    return _wrapped


def build_assistant_graph(ctx: NodeContext) -> Any:
    graph = StateGraph(GraphState)  # type: ignore[arg-type]

    graph.add_node("classify_interaction", classify_interaction)
    graph.add_node("normalise_input", normalise_input)
    graph.add_node("detect_ingestion_type", detect_ingestion_type)
    graph.add_node("transform_for_storage", transform_for_storage)
    graph.add_node("write_memobase", write_memobase(ctx))
    graph.add_node("write_vector_indexes", write_vector_indexes(ctx))
    graph.add_node("write_graph", write_graph(ctx))

    graph.add_node("classify_topic", classify_topic)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("route_retrieval", route_retrieval)
    graph.add_node("retrieve_memobase", _guarded(retrieve_memobase(ctx), "need_memobase"))
    graph.add_node("retrieve_vector", _guarded(retrieve_vector(ctx), "need_vector"))
    graph.add_node("retrieve_graph_pagerank", _guarded(retrieve_graph_pagerank(ctx), "need_graph"))
    graph.add_node("rerank", rerank)
    graph.add_node("tool_planner", tool_planner)
    graph.add_node("run_tools", run_tools(ctx))
    graph.add_node("ingest_tool_results", ingest_tool_results(ctx))
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_interaction")
    graph.add_conditional_edges(
        "classify_interaction",
        lambda state: state.get("interaction_kind", InteractionKind.ACT_AND_STORE).value,
        {
            InteractionKind.STORE_ONLY.value: "normalise_input",
            InteractionKind.TRANSFORM_AND_STORE.value: "normalise_input",
            InteractionKind.ACT_AND_STORE.value: "normalise_input",
            InteractionKind.ACT_NOW.value: "classify_topic",
        },
    )

    graph.add_edge("normalise_input", "detect_ingestion_type")
    graph.add_edge("detect_ingestion_type", "transform_for_storage")
    graph.add_edge("transform_for_storage", "write_memobase")
    graph.add_edge("write_memobase", "write_vector_indexes")
    graph.add_edge("write_vector_indexes", "write_graph")
    graph.add_conditional_edges(
        "write_graph",
        lambda state: state.get("interaction_kind", InteractionKind.ACT_AND_STORE).value,
        {
            InteractionKind.STORE_ONLY.value: "generate_answer",
            InteractionKind.TRANSFORM_AND_STORE.value: "generate_answer",
            InteractionKind.ACT_AND_STORE.value: "classify_topic",
            InteractionKind.ACT_NOW.value: "classify_topic",
        },
    )

    graph.add_edge("classify_topic", "classify_intent")
    graph.add_edge("classify_intent", "route_retrieval")
    graph.add_edge("route_retrieval", "retrieve_memobase")
    graph.add_edge("retrieve_memobase", "retrieve_vector")
    graph.add_edge("retrieve_vector", "retrieve_graph_pagerank")
    graph.add_edge("retrieve_graph_pagerank", "rerank")
    graph.add_edge("rerank", "tool_planner")
    graph.add_edge("tool_planner", "run_tools")
    graph.add_edge("run_tools", "ingest_tool_results")
    graph.add_edge("ingest_tool_results", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()

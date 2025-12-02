import asyncio

from daneel.clients import GraphClient, MemobaseClient, RagdollClient
from daneel.graph_builder import build_assistant_graph
from daneel.nodes import NodeContext


def test_graph_runs_end_to_end():
    ctx = NodeContext(
        memobase=MemobaseClient(),
        ragdoll=RagdollClient(),
        graph=GraphClient(),
        tools=None,  # type: ignore[arg-type]
    )
    graph = build_assistant_graph(ctx)
    initial_state = {"user_id": "tester", "query": "remember this task to email the team", "history": []}

    async def _run():
        if hasattr(graph, "ainvoke"):
            return await graph.ainvoke(initial_state)  # type: ignore[call-arg]
        return graph.invoke(initial_state)  # type: ignore[call-arg]

    final_state = asyncio.run(_run())
    assert final_state.get("answer")
    assert final_state.get("primary_topic")
    assert "tool_results" in final_state

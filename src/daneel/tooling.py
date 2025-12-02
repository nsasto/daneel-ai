from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional


ToolCallable = Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, ToolCallable] = {}

    def register(self, name: str, func: ToolCallable) -> None:
        self._registry[name] = func

    def get(self, name: str) -> Optional[ToolCallable]:
        return self._registry.get(name)

    async def run(self, name: str, **kwargs: Any) -> Any:
        func = self.get(name)
        if not func:
            raise KeyError(f"tool {name} not registered")
        return await func(**kwargs)


async def create_task_tool(title: str, **_: Any) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"status": "created", "title": title}


async def send_email_tool(body: str, **_: Any) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"status": "sent", "body": body}


async def schedule_meeting_tool(topic: str = "meeting", **_: Any) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"status": "scheduled", "topic": topic}


async def trigger_n8n_flow(flow_name: str = "default", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"status": "triggered", "flow": flow_name, "payload": payload or {}}


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("create_task", create_task_tool)
    registry.register("send_email", send_email_tool)
    registry.register("schedule_meeting", schedule_meeting_tool)
    registry.register("trigger_n8n_flow", trigger_n8n_flow)
    return registry

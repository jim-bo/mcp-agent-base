import json

import pytest

from base_agent.agent import ClaudeMCPAgent
from base_agent.mcp_client import EmptyMCPClient


class DummyToolDefinition:
    def __init__(self, name: str = "echo"):
        self.name = name
        self.description = "test tool"
        self.input_schema = {"type": "object"}

    def to_anthropic(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class DummyMCPClient:
    def __init__(self, server_url=None):
        self.server_url = server_url
        self.called = []

    async def list_tools(self):
        return [DummyToolDefinition()]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return json.dumps({"result": arguments})


class FakeDelta:
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


class FakeContentBlock:
    def __init__(self, block_type, text=None, block_id=None, name=None, input=None):
        self.type = block_type
        self.text = text
        self.id = block_id
        self.name = name
        self.input = input


class FakeEvent:
    def __init__(self, event_type, delta=None, content_block=None, index=None):
        self.type = event_type
        self.delta = delta
        self.content_block = content_block
        self.index = index


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeStream:
    def __init__(self, events, final_message):
        self._events = list(events)
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self):
        return self._final_message


class FakeAnthropic:
    def __init__(self, streams):
        self._streams = list(streams)

    @property
    def messages(self):
        return self

    def stream(self, **kwargs):
        if not self._streams:
            raise AssertionError("No prepared fake stream available")
        return self._streams.pop(0)


@pytest.mark.asyncio
async def test_agent_streams_tool_and_text(monkeypatch):
    # Avoid hitting a real MCP server by stubbing the client class used in __init__.
    monkeypatch.setattr("base_agent.agent.MCPClient", DummyMCPClient)

    tool_use_event = FakeEvent(
        "content_block_stop",
        content_block=FakeContentBlock(
            "tool_use", block_id="tool-1", name="echo", input={"x": 1}
        ),
    )
    first_stream = FakeStream([tool_use_event], FakeMessage([]))

    text_event = FakeEvent("content_block_delta", delta=FakeDelta("final text"))
    final_message = FakeMessage([FakeContentBlock("text", text="final text")])
    second_stream = FakeStream([text_event], final_message)

    agent = ClaudeMCPAgent(api_key="abc123", model="anthropic:claude-test", mock_empty_mcp=False)
    agent.client = FakeAnthropic([first_stream, second_stream])

    chunks = []
    async for chunk in agent.ask_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert any(isinstance(c, str) and "Calling `echo`" in c for c in chunks)
    assert any(
        isinstance(c, dict) and c.get("delta", {}).get("content") == "final text"
        for c in chunks
    )
    assert any(isinstance(c, dict) and c.get("finish_reason") == "stop" for c in chunks)
    assert agent.mcp_client.called == [("echo", {"x": 1})]
    # The model name should be normalized to drop provider prefixes.
    assert agent.model == "claude-test"


@pytest.mark.asyncio
async def test_agent_ask_collects_tool_logs(monkeypatch):
    monkeypatch.setattr("base_agent.agent.MCPClient", DummyMCPClient)

    tool_use_event = FakeEvent(
        "content_block_stop",
        content_block=FakeContentBlock(
            "tool_use", block_id="tool-1", name="echo", input={"y": 2}
        ),
    )
    first_stream = FakeStream([tool_use_event], FakeMessage([]))
    final_message = FakeMessage([FakeContentBlock("text", text="complete")])
    second_stream = FakeStream([], final_message)

    agent = ClaudeMCPAgent(api_key="abc123", model="claude-test", mock_empty_mcp=False)
    agent.client = FakeAnthropic([first_stream, second_stream])

    answer = await agent.ask([{"role": "user", "content": "hi"}])

    assert "Calling `echo`" in answer
    assert "Result from `echo`" in answer
    assert "complete" in answer


@pytest.mark.asyncio
async def test_agent_simulate_streams_canned_response(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = ClaudeMCPAgent(simulate=True)

    chunks = []
    async for chunk in agent.ask_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert any(chunk.get("delta", {}).get("content") == "Simulated response." for chunk in chunks if isinstance(chunk, dict))
    assert any(chunk.get("finish_reason") == "stop" for chunk in chunks if isinstance(chunk, dict))
    assert agent.client is None
    assert agent.mcp_client is None


@pytest.mark.asyncio
async def test_agent_simulate_non_stream(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = ClaudeMCPAgent(simulate=True)

    result = await agent.ask([{"role": "user", "content": "hi"}])
    assert result == "Simulated response."


@pytest.mark.asyncio
async def test_agent_streams_thinking_separately(monkeypatch):
    monkeypatch.setattr("base_agent.agent.MCPClient", DummyMCPClient)

    events = [
        FakeEvent("content_block_start", content_block=FakeContentBlock("thinking"), index=0),
        FakeEvent("content_block_delta", delta=FakeDelta("inner thought"), index=0),
        FakeEvent("content_block_stop", index=0),
        FakeEvent("content_block_start", content_block=FakeContentBlock("text"), index=1),
        FakeEvent("content_block_delta", delta=FakeDelta("final answer"), index=1),
        FakeEvent("content_block_stop", index=1),
    ]
    final_message = FakeMessage([FakeContentBlock("text", text="final answer")])
    stream = FakeStream(events, final_message)

    agent = ClaudeMCPAgent(api_key="abc123", model="claude-test")
    agent.client = FakeAnthropic([stream])

    chunks = []
    async for chunk in agent.ask_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert any(
        isinstance(chunk, dict) and chunk.get("delta", {}).get("thinking") == "inner thought"
        for chunk in chunks
    ), "expected thinking delta"
    assert any(
        isinstance(chunk, dict)
        and chunk.get("delta", {}).get("content") == "inner thought"
        for chunk in chunks
    ), "thinking should surface in content deltas when present"
    assert any(
        isinstance(chunk, dict) and chunk.get("delta", {}).get("content") == "final answer"
        for chunk in chunks
    ), "expected normal content delta"


@pytest.mark.asyncio
async def test_agent_allows_mock_empty_mcp(monkeypatch):
    # Fake anthropic stream that only returns final text; no tool use.
    events = [FakeEvent("content_block_delta", delta=FakeDelta("final answer"), index=0)]
    final_message = FakeMessage([FakeContentBlock("text", text="final answer")])
    stream = FakeStream(events, final_message)

    agent = ClaudeMCPAgent(api_key="abc123", model="claude-test", mock_empty_mcp=True)
    agent.client = FakeAnthropic([stream])

    chunks = []
    async for chunk in agent.ask_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert any(isinstance(c, dict) and c.get("delta", {}).get("content") == "final answer" for c in chunks)

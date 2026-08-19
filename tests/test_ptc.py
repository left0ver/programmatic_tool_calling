import json
from types import SimpleNamespace

import pytest

from ptc import (
    CodeRejectedError,
    OpenAIProgrammaticAgent,
    PTCError,
    RestrictedPythonExecutor,
    StepLimitExceeded,
    ToolRegistry,
    schema_from_callable,
)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @registry.tool()
    def prices(category: str) -> list[dict]:
        """Return product prices."""
        return [{"category": category, "price": 5}, {"category": category, "price": 8}]

    return registry


def test_schema_is_inferred_from_signature() -> None:
    def lookup(query: str, limit: int = 10) -> list[dict]:
        return []

    schema = schema_from_callable(lookup)
    assert schema["properties"]["query"] == {"type": "string"}
    assert schema["properties"]["limit"] == {"type": "integer", "default": 10}
    assert schema["required"] == ["query"]


def test_prompt_publishes_input_and_output_contracts() -> None:
    prompt = make_registry().prompt()
    assert '"a": {"type": "integer"}' in prompt
    assert 'returns: {"type": "integer"}' in prompt


def test_executes_multiple_tools_and_returns_only_result() -> None:
    execution = RestrictedPythonExecutor(make_registry()).execute(
        """
base = add(2, 3)
items = prices("books")
result = {"total": base + sum(item["price"] for item in items)}
"""
    )
    assert execution.result == {"total": 18}
    assert [call.name for call in execution.tool_calls] == ["add", "prices"]


@pytest.mark.parametrize(
    "source",
    ["import os\nresult = 1", "result = (1).__class__", "_secret = 1\nresult = 1"],
)
def test_rejects_unsafe_syntax(source: str) -> None:
    with pytest.raises(CodeRejectedError):
        RestrictedPythonExecutor(make_registry()).execute(source)


def test_requires_result_variable() -> None:
    with pytest.raises(PTCError, match="must assign"):
        RestrictedPythonExecutor(make_registry()).execute("value = add(1, 2)")


def test_stops_runaway_program() -> None:
    with pytest.raises(StepLimitExceeded):
        RestrictedPythonExecutor(make_registry(), max_steps=20).execute(
            "while True:\n    value = 1\nresult = value"
        )


def test_agent_returns_execution_result_to_model() -> None:
    class Item(SimpleNamespace):
        def model_dump(self, **kwargs):
            return vars(self)

    class Responses:
        def __init__(self) -> None:
            self.requests = []

        def create(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                call = Item(
                    type="function_call",
                    name="execute_python",
                    call_id="call-1",
                    arguments=json.dumps({"code": "result = add(20, 22)"}),
                )
                return SimpleNamespace(output=[call], output_text="")
            return SimpleNamespace(
                output=[Item(type="message", content=[])], output_text="The answer is 42."
            )

    responses = Responses()
    client = SimpleNamespace(responses=responses)
    answer = OpenAIProgrammaticAgent(client, "test-model", make_registry()).run("add")

    assert answer == "The answer is 42."
    tool_output = next(
        item
        for item in responses.requests[1]["input"]
        if item.get("type") == "function_call_output"
    )
    assert tool_output["type"] == "function_call_output"
    assert json.loads(tool_output["output"])["result"] == 42

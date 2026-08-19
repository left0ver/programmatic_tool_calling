"""A small, generic Programmatic Tool Calling runtime for Python."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import wraps
from types import UnionType
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

from openai import OpenAI
from RestrictedPython.compile import compile_restricted_exec
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector


class PTCError(RuntimeError):
    """Base error raised by the local PTC runtime."""


class CodeRejectedError(PTCError):
    """The generated program does not belong to the allowed Python subset."""


class StepLimitExceeded(PTCError):
    """The generated program used more Python execution steps than allowed."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    output_schema: dict[str, Any]
    function: Callable[..., Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ExecutionResult:
    result: Any
    printed: str
    tool_calls: list[ToolCall]

    def model_payload(self) -> str:
        return json.dumps(
            {"ok": True, **asdict(self)}, ensure_ascii=False, separators=(",", ":")
        )


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if values:
            schema["type"] = _annotation_schema(type(values[0])).get("type")
        return schema
    if origin in (Union, UnionType):
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1 and len(non_null) != len(args):
            return {"anyOf": [_annotation_schema(non_null[0]), {"type": "null"}]}
        return {"anyOf": [_annotation_schema(item) for item in args]}
    if origin in (list, tuple, set):
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {}}
    if origin is dict:
        return {
            "type": "object",
            "additionalProperties": _annotation_schema(args[1])
            if len(args) > 1
            else True,
        }
    primitive = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if annotation in primitive:
        return {"type": primitive[annotation]}
    return {}


def schema_from_callable(function: Callable[..., Any]) -> dict[str, Any]:
    """Create a practical JSON schema from a function's typed signature."""
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError(f"tool {function.__name__!r} cannot use *args or **kwargs")
        properties[name] = _annotation_schema(hints.get(name, parameter.annotation))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def output_schema_from_callable(function: Callable[..., Any]) -> dict[str, Any]:
    """Infer the output schema when a tool has a useful return annotation."""
    hints = get_type_hints(function)
    return _annotation_schema(hints.get("return", inspect.Signature.empty))


class ToolRegistry:
    """Registers arbitrary Python callables and publishes them to the code model."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        tool_name = name or function.__name__
        if not tool_name.isidentifier() or tool_name.startswith("_"):
            raise ValueError(f"invalid Python tool name: {tool_name!r}")
        if tool_name in self._tools:
            raise ValueError(f"duplicate tool name: {tool_name}")
        tool_description = description or inspect.getdoc(function)
        if not tool_description:
            raise ValueError(f"tool {tool_name!r} needs a description or docstring")
        self._tools[tool_name] = Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters or schema_from_callable(function),
            output_schema=output_schema or output_schema_from_callable(function),
            function=function,
        )
        return function

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                function,
                name=name,
                description=description,
                parameters=parameters,
                output_schema=output_schema,
            )
            return function

        return decorator

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def prompt(self) -> str:
        lines = ["Available Python tool functions:"]
        for tool in self.tools:
            input_schema = json.dumps(tool.parameters, ensure_ascii=False)
            output_schema = json.dumps(tool.output_schema, ensure_ascii=False)
            lines.append(
                f"- {tool.name}(**arguments): {tool.description}"
                f"\n  arguments: {input_schema}\n  returns: {output_schema}"
            )
        return "\n".join(lines)


def _json_value(value: Any, *, max_bytes: int) -> Any:
    """Detach tool output from arbitrary Python objects before exposing it to code."""
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise PTCError(f"tool returned a non-JSON value: {exc}") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise PTCError(f"tool result exceeded {max_bytes} bytes")
    return json.loads(encoded)


class RestrictedPythonExecutor:
    """Compile and execute model-generated Python with an explicit policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_code_chars: int = 20_000,
        max_steps: int = 100_000,
        max_result_bytes: int = 1_000_000,
    ) -> None:
        self.registry = registry
        self.max_code_chars = max_code_chars
        self.max_steps = max_steps
        self.max_result_bytes = max_result_bytes

    def _validate_source(self, source: str) -> None:
        if len(source) > self.max_code_chars:
            raise CodeRejectedError("generated code is too large")
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise CodeRejectedError(str(exc)) from exc
        banned = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)
        for node in ast.walk(tree):
            if isinstance(node, banned):
                raise CodeRejectedError(f"{type(node).__name__} is not allowed")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise CodeRejectedError("names beginning with '_' are not allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise CodeRejectedError("private attributes are not allowed")

    def execute(self, source: str) -> ExecutionResult:
        self._validate_source(source)
        compiled = compile_restricted_exec(source, filename="<ai-generated-ptc>")
        if compiled.errors:
            raise CodeRejectedError("; ".join(compiled.errors))
        calls: list[ToolCall] = []
        exposed_tools = {
            tool.name: self._wrap_tool(tool, calls) for tool in self.registry.tools
        }
        builtins = dict(safe_builtins)
        builtins.update(
            {
                "all": all,
                "any": any,
                "enumerate": enumerate,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "range": range,
                "reversed": reversed,
                "sorted": sorted,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            }
        )
        globals_: dict[str, Any] = {
            "__builtins__": builtins,
            "_getattr_": safer_getattr,
            "_getitem_": lambda value, key: value[key],
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_write_": full_write_guard,
            "_print_": PrintCollector,
            **exposed_tools,
        }
        locals_: dict[str, Any] = {}
        steps = 0

        def trace(frame: Any, event: str, arg: Any) -> Callable[..., Any]:
            nonlocal steps
            if event == "line" and frame.f_code.co_filename == "<ai-generated-ptc>":
                steps += 1
                if steps > self.max_steps:
                    raise StepLimitExceeded(
                        f"generated code exceeded {self.max_steps} execution steps"
                    )
            return trace

        old_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            exec(compiled.code, globals_, locals_)
        finally:
            sys.settrace(old_trace)
        if "result" not in locals_:
            raise PTCError("generated code must assign its final value to `result`")
        result = _json_value(locals_["result"], max_bytes=self.max_result_bytes)
        printed = str(locals_.get("printed", ""))
        return ExecutionResult(result=result, printed=printed, tool_calls=calls)

    def _wrap_tool(self, tool: Tool, calls: list[ToolCall]) -> Callable[..., Any]:
        @wraps(tool.function)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            signature = inspect.signature(tool.function)
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as exc:
                raise PTCError(f"invalid arguments for {tool.name}: {exc}") from exc
            arguments = dict(bound.arguments)
            calls.append(
                ToolCall(
                    tool.name, _json_value(arguments, max_bytes=self.max_result_bytes)
                )
            )
            return _json_value(
                tool.function(*args, **kwargs), max_bytes=self.max_result_bytes
            )

        return invoke


class ProgrammaticAgent:
    """An OpenAI Responses API loop whose single model tool executes Python."""

    EXECUTE_TOOL: ClassVar[dict[str, Any]] = {
        "type": "function",
        "name": "execute_python",
        "description": "Execute a RestrictedPython program that may call the listed tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source. Assign the compact final value to `result`.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    def __init__(
        self,
        client: OpenAI,
        model: str,
        registry: ToolRegistry,
        *,
        max_rounds: int = 6,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = RestrictedPythonExecutor(registry)
        self.max_rounds = max_rounds

    def run(self, query: str, *, verbose: bool = False) -> str:
        instructions = f"""You solve requests with programmatic tool calling.
Call execute_python when tools or deterministic data processing are needed. The code is
Python, cannot import modules, and must assign a JSON-compatible final value to `result`.
Do filtering, joins, loops, arithmetic, and aggregation in that one program so only the
compact final result returns to you. Never invent tool results. If execution returns an error, correct the code and retry.

{self.registry.prompt()}
"""
        items: list[Any] = [{"role": "user", "content": query}]
        for round_number in range(1, self.max_rounds + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=items,
                tools=[self.EXECUTE_TOOL],
                store=False,
            )
            items.extend(item.model_dump(exclude_none=True) for item in response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            if verbose:
                print(f"\n--- model round {round_number} ---")
                print(response.output_text or f"{len(calls)} function call(s)")
            if not calls:
                if response.output_text:
                    return response.output_text
                raise PTCError("model returned neither a function call nor an answer")
            for call in calls:
                if call.name != "execute_python":
                    payload = json.dumps({"ok": False, "error": "unknown model tool"})
                else:
                    code = json.loads(call.arguments)["code"]
                    if verbose:
                        print(f"\nGenerated Python:\n{code}")
                    try:
                        execution = self.executor.execute(code)
                        payload = execution.model_payload()
                    except Exception as exc:
                        payload = json.dumps(
                            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                            ensure_ascii=False,
                        )
                    if verbose:
                        print(f"\nExecution output:\n{payload}")
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": payload,
                    }
                )
        raise PTCError(f"model did not finish after {self.max_rounds} rounds")

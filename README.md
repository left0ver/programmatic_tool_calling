# 通用 Python Programmatic Tool Calling

一个本地 Python 版的 Programmatic Tool Calling（PTC）运行时，参考 [programmatic_tool_calling](https://github.com/29swastik/programmatic_tool_calling) 实现。应用将 Python 函数注册为工具；模型不直接逐个调用这些工具，而是调用唯一的 `execute_python`，在受限 Python 环境中编写一段可组合多个工具的程序。

与传统 tool calling 相比，筛选、循环、聚合和多工具数据关联可以在同一段程序内完成，避免把每个中间工具结果逐轮发送回模型。

## 工作方式

1. `ToolRegistry` 根据函数签名、类型标注和 docstring 发布工具说明及 JSON Schema。
2. `ProgrammaticAgent` 使用 OpenAI Responses API，并只向模型提供 `execute_python` 一个函数工具。
3. 模型生成 Python 代码，在代码中调用已注册工具，并把最终 JSON 兼容值赋给 `result`。
4. `RestrictedPythonExecutor` 校验、编译并执行代码；工具的参数和返回值都会经过 JSON 往返转换。
5. 执行结果（`result`、记录到的工具调用和 `printed` 内容）作为函数调用输出返回模型；模型随后生成最终回答。执行失败时，模型可以在后续轮次修正代码并重试。

默认最多执行 6 个模型轮次；超过后会抛出 `PTCError`。

## 环境要求与安装

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- 支持 OpenAI Responses API 的服务及其 API 密钥

```bash
cp .env.example .env
uv sync
```

在 `.env` 中配置：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-5.6
```

`LLM_BASE_URL` 可指向实现了 OpenAI Responses API 的兼容服务。

## 运行旅行示例

项目内的 [travel_example.py](travel_example.py) 注册航班、酒店和天气工具，让模型在一次受限 Python 执行中筛选三个城市并计算旅行总价。

```bash
uv run python travel_example.py
```

示例以 `verbose=True` 运行，会打印每轮模型响应、模型生成的 Python 代码和执行输出，便于调试。

## 接入自己的工具

为工具提供类型标注和 docstring，即可自动生成参数 Schema：

```python
from openai import OpenAI

from ptc import ProgrammaticAgent, ToolRegistry

registry = ToolRegistry()


@registry.tool()
def search_orders(customer_id: str, limit: int = 20) -> list[dict]:
    """Return recent orders with `order_id` and numeric `amount` fields."""
    return your_database_query(customer_id, limit)


client = OpenAI()
agent = ProgrammaticAgent(client, "gpt-5.6", registry)
print(agent.run("统计客户 c-123 最近订单的总金额"))
```

`@registry.tool()` 也支持 `name`、`description`、`parameters` 和 `output_schema` 参数。复杂入参或结构化返回值可显式提供 JSON Schema：

```python
registry.register(
    search_orders,
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
    },
)
```

工具名必须是非下划线开头的合法 Python 标识符，且不能重复；函数不能使用 `*args` 或 `**kwargs`。工具参数、工具返回值和最终 `result` 都必须可 JSON 序列化。为字典或列表中的对象补充 `output_schema`，能让模型更可靠地编写筛选和聚合代码。

## 直接使用执行器

`RestrictedPythonExecutor` 可独立用于测试模型生成的代码。代码必须为最终值赋给 `result`；执行结果同时提供 `result`、`printed` 和 `tool_calls`。

```python
from ptc import RestrictedPythonExecutor

execution = RestrictedPythonExecutor(registry).execute("""
orders = search_orders(customer_id="c-123", limit=10)
result = sum(order["amount"] for order in orders)
""")
print(execution.result)
print(execution.tool_calls)
```

执行器默认限制为 20,000 个源代码字符、100,000 个 Python 执行步数和 1 MB 的单个 JSON 工具结果/最终结果。可在构造 `RestrictedPythonExecutor` 时通过 `max_code_chars`、`max_steps`、`max_result_bytes` 调整。

## 测试

```bash
uv run pytest
```

## 安全边界

`RestrictedPython` 官方明确说明它不是完整沙箱。本项目额外禁止 import、私有名称访问、`global`/`nonlocal` 和类定义，只暴露白名单 builtins，并限制源码、执行步数及 JSON 结果大小。这适合可信应用内由模型生成的胶水代码，不应作为多租户恶意代码隔离方案。

生产环境还应：

- 在独立容器或微虚机中运行代码，并设置 CPU、内存、网络和文件系统限制；
- 对有副作用的工具单独授权，默认不要向 PTC 暴露写操作；
- 在每个工具内部完成身份验证、参数校验、超时、限流和审计；
- 注意执行步数限制不能中断一个已进入阻塞状态的工具函数。

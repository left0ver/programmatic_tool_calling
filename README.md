# 通用 Python Programmatic Tool Calling

这个项目参考 [programmatic_tool_calling](https://github.com/29swastik/programmatic_tool_calling)，实现了一个本地 Python 版 PTC：

1. 应用注册任意 Python 工具；
2. 模型看到自动生成的工具说明，但只调用一个 `execute_python`；
3. 模型生成可串联多个工具的 Python 程序；
4. `RestrictedPython` 编译并在白名单环境中执行程序；
5. 只有程序写入 `result` 的精简结果会返回给模型生成自然语言答案。

相比传统 tool calling，每个中间工具结果不必逐轮发回模型。

## 运行旅行示例

```bash
cp .env.example .env
uv sync
uv run python ptc.py
```

`.env`：

```dotenv
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-5.6
```

`LLM_BASE_URL` 也可以指向实现 OpenAI Responses API 的兼容服务。

## 注册任意工具

最简单的方式是提供类型标注和 docstring，参数 JSON Schema 会自动生成：

```python
from ptc import OpenAIProgrammaticAgent, ToolRegistry

registry = ToolRegistry()

@registry.tool()
def search_orders(customer_id: str, limit: int = 20) -> list[dict]:
    """Return recent orders with `order_id` and numeric `amount` fields."""
    return your_database_query(customer_id, limit)

agent = OpenAIProgrammaticAgent(client, "gpt-5.6", registry)
print(agent.run("统计客户 c-123 最近订单的总金额"))
```

复杂参数和结构化返回值可以通过 `registry.register(..., parameters={...}, output_schema={...})` 显式传入 JSON Schema。建议为字典返回值明确描述字段，否则模型无法可靠地编写筛选代码。工具的参数和返回值必须可 JSON 序列化。执行器会对返回值做 JSON 往返转换，模型代码不能获得数据库连接、SDK 客户端等原始对象。

## 直接测试执行器

```python
from ptc import RestrictedPythonExecutor

execution = RestrictedPythonExecutor(registry).execute("""
orders = search_orders(customer_id="c-123", limit=10)
result = sum(order["amount"] for order in orders)
""")
print(execution.result)
print(execution.tool_calls)
```

```bash
uv run pytest
```

## 安全边界

`RestrictedPython` 官方明确说明它不是完整沙箱。这里额外关闭 import、私有名称访问和危险 builtins，限制源码大小、执行步数和 JSON 结果大小；这适合可信应用内由模型生成的胶水代码，但不应作为多租户恶意代码隔离方案。

生产环境还应：

- 在独立容器或微虚机中运行代码，并设置 CPU、内存、网络和文件系统限制；
- 对有副作用的工具单独授权，写操作默认不要暴露给 PTC；
- 在每个工具内部做身份验证、参数校验、超时、限流和审计；
- 注意执行步数限制无法中断一个已经进入阻塞状态的工具函数。

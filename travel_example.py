"""Travel example built on the generic local Python PTC runtime."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from ptc import ProgrammaticAgent, ToolRegistry

load_dotenv()
registry = ToolRegistry()


@registry.tool(
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "flight": {"type": "string"},
                "price": {"type": "number"},
                "direct": {"type": "boolean"},
                "seats": {"type": "integer"},
            },
            "required": ["flight", "price", "direct", "seats"],
        },
    }
)
def search_flights(city: str) -> list[dict]:
    """Return flight options for a destination city."""
    data = {
        "上海": [
            {"flight": "MU5101", "price": 980, "direct": True, "seats": 4},
            {"flight": "HO1203", "price": 760, "direct": False, "seats": 8},
        ],
        "成都": [
            {"flight": "CA1407", "price": 820, "direct": True, "seats": 3},
            {"flight": "3U8882", "price": 690, "direct": True, "seats": 0},
        ],
        "西安": [
            {"flight": "HU7137", "price": 720, "direct": True, "seats": 5},
            {"flight": "MU2118", "price": 640, "direct": True, "seats": 2},
        ],
    }
    return data[city]


@registry.tool(
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "hotel": {"type": "string"},
                "nightly_price": {"type": "number"},
                "rating": {"type": "number"},
            },
            "required": ["hotel", "nightly_price", "rating"],
        },
    }
)
def search_hotels(city: str) -> list[dict]:
    """Return hotel options for a destination city."""
    data = {
        "上海": [
            {"hotel": "外滩酒店", "nightly_price": 520, "rating": 4.6},
            {"hotel": "静安旅店", "nightly_price": 420, "rating": 4.3},
        ],
        "成都": [
            {"hotel": "春熙酒店", "nightly_price": 380, "rating": 4.7},
            {"hotel": "锦里旅店", "nightly_price": 320, "rating": 4.4},
        ],
        "西安": [
            {"hotel": "城墙酒店", "nightly_price": 360, "rating": 4.5},
            {"hotel": "钟楼旅店", "nightly_price": 310, "rating": 4.2},
        ],
    }
    return data[city]


@registry.tool(
    output_schema={
        "type": "object",
        "properties": {"rainy_days": {"type": "integer"}},
        "required": ["rainy_days"],
    }
)
def get_weather(city: str) -> dict:
    """Return the number of rainy days in the three-day forecast."""
    return {
        "上海": {"rainy_days": 2},
        "成都": {"rainy_days": 1},
        "西安": {"rainy_days": 0},
    }[city]


def main() -> None:

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL"),
    )
    agent = ProgrammaticAgent(
        client=client,
        model=os.environ["LLM_MODEL_NAME"],
        registry=registry,
    )
    answer = agent.run(
        "为上海、成都、西安选择三日旅行目的地：只考虑最多一个雨天；选择有座位的最便宜直飞航班，"
        "以及评分至少 4.5 的最便宜酒店；按航班加两晚酒店的总价排序。",
        verbose=True,
    )
    print(f"\nFinal answer:\n{answer}")


if __name__ == "__main__":
    main()

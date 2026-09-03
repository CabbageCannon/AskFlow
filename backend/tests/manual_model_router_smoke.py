"""Manual smoke test for the dynamic model router.

This test performs a REAL Alibaba Bailian API call.
Do not include it in normal automated pytest runs.
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

import open_deep_research.deep_researcher as dr


async def main():
    # 普通 python 脚本不会像 langgraph dev 一样自动帮我们加载 .env，
    # 因此这里显式加载。
    load_dotenv()

    configurable = dr.Configuration.from_runnable_config({})

    if not configurable.bailian_api_key:
        raise RuntimeError(
            "BAILIAN_API_KEY is missing. "
            "Please configure it in backend/.env first."
        )

    print("[SMOKE_TEST] configuration loaded")
    print(
        "[SMOKE_TEST] base_url="
        f"{configurable.bailian_base_url}"
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "请研究 DeepSeek 当前 API 的主要模型、"
                    "上下文窗口和 Tool Calling 支持情况，"
                    "并优先使用官方资料。"
                )
            )
        ]
    }

    print("[SMOKE_TEST] invoking clarify_with_user...")

    result = await dr.clarify_with_user(
        state,
        {},
    )
    
    clarification_update = result.update or {}

    clarification_messages = clarification_update.get(
        "messages",
        [],
    )

    brief_state = {
        "messages": [
            *state["messages"],
            *clarification_messages,
        ]
    }

    print()
    print("[SMOKE_TEST] invoking write_research_brief...")

    brief_result = await dr.write_research_brief(
        brief_state,
        {},
    )

    print()
    print("[SMOKE_TEST] research brief success")
    print(
        "[SMOKE_TEST] goto="
        f"{brief_result.goto}"
    )

    brief_update = brief_result.update or {}

    print(
        "[SMOKE_TEST] research_brief="
        f"{brief_update.get('research_brief')}"
    )

    print()
    print("[SMOKE_TEST] success")
    print(f"[SMOKE_TEST] goto={result.goto}")

    update = result.update or {}

    messages = update.get(
        "messages",
        [],
    )

    for message in messages:
        print(
            "[SMOKE_TEST] assistant_message="
            f"{message.content}"
        )


if __name__ == "__main__":
    asyncio.run(main())
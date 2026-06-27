import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.uuid import uuid7
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

from model.llm import QwenConfig

PROMPT = "What's the latest LLM released by Anthropic?"


def build_agent():
    config = QwenConfig(
        model="qwen3.6-max-preview",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    model = init_chat_model(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        model_provider="openai",
    )

    search_tool = TavilySearch(max_results=5)

    return create_agent(
        model=model,
        tools=[search_tool],
        checkpointer=InMemorySaver(),
    )

def query_streaming(prompt: str) -> None:
    agent = build_agent()

    config: RunnableConfig = {
        "configurable": {
            "thread_id": str(uuid7())
        }
    }

    for mode, data in agent.stream(
        input={"messages": HumanMessage(prompt)},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, _metadata = data
            if chunk.content:
                if isinstance(chunk, AIMessage):
                    print(chunk.content, end="", flush=True)
                elif isinstance(chunk, ToolMessage):
                    print(f"Tool message: {chunk.content}")
                continue

            if chunk.tool_calls:
                print(f"\ntool_calls")
            #     print(f"\nchunk: {chunk}")
            #     print(f"\nCalling tools: {[tc['name'] for tc in chunk.tool_calls]}")
        elif mode == "updates":
            for node_name, update in data.items():
                if node_name == "model":
                    last_msg = update["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            print(f"🔧 **Calling `{tc['name']}`**")
                            print(tc["args"])
                elif node_name == "tools":
                    for msg in update["messages"]:
                        if isinstance(msg, ToolMessage):
                            content = msg.content
                            if isinstance(content, list):
                                content = str(content)
                            if isinstance(content, str) and len(content) > 500:
                                content = content[:500] + "..."

                            print(f"✅ **`{msg.name}`** returned:")
                            print(content)


def main() -> None:
    load_dotenv()
    query_streaming(PROMPT)


if __name__ == "__main__":
    main()

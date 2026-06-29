import os
from datetime import datetime

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch

from model.llm import QwenConfig


def main() -> None:
    load_dotenv()
    config = QwenConfig(
        model="qwen3.7-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    llm = init_chat_model(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        model_provider="openai",
    )

    agent = create_deep_agent(
        model=llm,
        tools=[TavilySearch(max_results=5)],
        system_prompt=f"You are a helpful assistant. Currently it's {datetime.now().year} this year, please offer your answers based on that.",
    )

    for chunk, _metadata in agent.stream(
        input={"messages": [HumanMessage("What is the latest released LLM in Anthropic?")]},
        stream_mode="messages",
    ):
        if chunk.content:
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, ToolMessage):
                print("\n" + "=*" * 20)
                print(f"Tool message:\n{chunk.content}")
                print("=*" * 20 + "\n")

if __name__ == "__main__":
    main()

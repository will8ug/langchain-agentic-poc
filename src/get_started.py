import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

from model.llm import QwenConfig

load_dotenv()

qwen = QwenConfig(
    model="qwen3.6-max-preview",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

model = init_chat_model(
    model=qwen.model,
    api_key=qwen.api_key,
    base_url=qwen.base_url,
    model_provider="openai",
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}
for chunk in agent.stream(
    input={"messages": HumanMessage("Tell me a joke")},
    config=config,
    stream_mode="values"
):
    last_msg = chunk["messages"][-1]
    if last_msg.content:
        if isinstance(last_msg, AIMessage):
            print(last_msg.content, end="", flush=True)
        continue

    if last_msg.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in last_msg.tool_calls]}")

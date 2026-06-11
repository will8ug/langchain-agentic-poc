import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = init_chat_model(
    model="qwen3.6-max-preview",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai"
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
response = agent.invoke(
    input={"messages": HumanMessage("What is the capital of France?")},
    config=config,
)
print(response["messages"][-1].content)

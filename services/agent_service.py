import streamlit as st
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from services.tools import web_search_tool


@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    agent = research_agent()
    result = agent.invoke(query)
    return result["messages"][-1].content


@st.cache_resource
def get_checkpointer():
    return MemorySaver()

@st.cache_resource
def get_main_agent(model: str):
    llm = init_chat_model(
        model=model,
        api_key=st.secrets["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
    )
    return create_agent(model=llm, tools=[call_research_agent], checkpointer=get_checkpointer())

@st.cache_resource
def research_agent():
    llm = init_chat_model(
        model="qwen3.6-plus-2026-04-02",
        api_key=st.secrets["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
    )
    return create_agent(model=llm, tools=[web_search_tool()])

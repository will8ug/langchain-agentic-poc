import functools
import os
from datetime import datetime

import streamlit as st
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from services.tools import web_search_tool, get_current_date


@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    agent = research_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
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
    return create_agent(
        model=llm,
        tools=[get_current_date, call_research_agent],
        system_prompt=f"You are a helpful assistant. Currently it's {datetime.now().year} this year, please offer your answers based on that.",
        checkpointer=get_checkpointer()
    )

@functools.lru_cache(maxsize=1)
def research_agent():
    llm = init_chat_model(
        model="qwen3.6-plus-2026-04-02",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
    )
    return create_agent(model=llm, tools=[web_search_tool()])

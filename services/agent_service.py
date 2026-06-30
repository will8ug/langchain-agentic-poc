import functools
import os
from datetime import datetime

import streamlit as st
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from services.tools import web_search_tool, get_current_date


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
    return create_deep_agent(
        model=llm,
        tools=[get_current_date],
        system_prompt=f"You are a helpful assistant. Currently it's {datetime.now().year} this year, please offer your answers based on that.",
        subagents=[research_agent()],
        checkpointer=get_checkpointer()
    )

@st.cache_resource
def research_agent() -> CompiledSubAgent:
    llm = init_chat_model(
        model="qwen3.6-plus-2026-04-02",
        api_key=st.secrets["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
    )
    return {
        "name": "research-agent",
        "description": "Used to research more in depth questions",
        "runnable": create_agent(
            model=llm,
            tools=[web_search_tool()]
        )
    }

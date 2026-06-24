import os

import streamlit as st
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver


@st.cache_resource
def get_checkpointer():
    return MemorySaver()


@st.cache_resource
def get_agent(model: str):
    llm = init_chat_model(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
    )
    return create_agent(model=llm, checkpointer=get_checkpointer())

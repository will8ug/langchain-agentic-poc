import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from services.agent_service import get_agent
from ui.components.sidebar import sidebar


def render_chat_history(messages: list[BaseMessage]):
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


st.title("💬 AI Chat Assistant")

model_name= sidebar()
config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
agent = get_agent(model=model_name)

render_chat_history(agent.get_state(config).values.get("messages", []))

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        try:
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(content=str(prompt))]},
                config=config,
                stream_mode="messages",
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_box.markdown(full_response)

            response_box.markdown(full_response)

        except Exception as e:
            st.error(f"❌ Error: {e}")

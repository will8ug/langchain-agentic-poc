import logging

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from services.agent_service import get_main_agent
from ui.components.sidebar import sidebar


def render_chat_history(messages: list[BaseMessage]) -> None:
    tool_results: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg

    for msg in messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                if msg.content:
                    st.markdown(msg.content)
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    label = f"🔧 Tool Calls ({len(msg.tool_calls)})"
                    with st.expander(label, expanded=False):
                        for tc in msg.tool_calls:
                            st.markdown(f"**Called `{tc['name']}`**")
                            st.json(tc["args"])
                            result = tool_results.get(tc.get("id") or "")
                            if result:
                                st.markdown(f"**Result:**")
                                st.text(result.content)
        # ToolMessages are rendered inside the preceding AIMessage's expander


def _stream_message_chunk(chunk, response_box, full_response: str) -> str:
    if chunk.content:
        full_response += chunk.content
        response_box.markdown(full_response + " ▌")
    return full_response


def _stream_update(data, status) -> bool:
    has_tool_calls = False
    for node_name, update in data.items():
        if node_name == "model":
            last_msg = update["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                has_tool_calls = True
                for tc in last_msg.tool_calls:
                    with status:
                        st.markdown(f"🔧 **Calling `{tc['name']}`**")
                        st.json(tc["args"])
        elif node_name == "tools":
            for msg in update["messages"]:
                if isinstance(msg, ToolMessage):
                    content = msg.content
                    if isinstance(content, list):
                        content = str(content)
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + "..."
                    with status:
                        st.markdown(f"✅ **`{msg.name}`** returned:")
                        st.text(content)
    return has_tool_calls


st.title("💬 AI Chat Assistant")

model_name = sidebar()
config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
agent = get_main_agent(model=model_name)

render_chat_history(agent.get_state(config).values.get("messages", []))

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        response_box = st.empty()
        full_response = ""
        has_tool_calls = False

        try:
            # stream_mode as a list yields (mode, data) tuples:
            #   "messages" -> data is (chunk, metadata)
            #   "updates"  -> data is {node_name: state_update_dict}
            for mode, data in agent.stream(
                {"messages": [HumanMessage(content=str(prompt))]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages" and isinstance(data, tuple):
                    chunk, _metadata = data
                    full_response = _stream_message_chunk(chunk, response_box, full_response)

                elif mode == "updates" and isinstance(data, dict):
                    has_tool_calls = _stream_update(data, status) or has_tool_calls

            response_box.markdown(full_response)

            status.update(
                label="Tool calls completed" if has_tool_calls else "Done",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            logging.exception("Agent stream failed")
            status.update(label="Error", state="error")
            st.error(f"❌ Error: {e}")

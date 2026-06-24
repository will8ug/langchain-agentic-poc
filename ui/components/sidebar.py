import uuid

import streamlit as st


def sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox(
            "Model",
            [
                "qwen3.6-flash",
                "deepseek-v4-flash",
                "qwen3.6-max-preview",
                "qwen3.7-max-preview",
                "qwen3.7-plus",
            ],
            index=0,
        )

        if st.button("🗑️ Clear History"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    return model_name

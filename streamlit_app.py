import uuid

import streamlit as st

st.set_page_config(page_title="🤖 LangChain + Streamlit", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

page = st.Page("app_pages/chat.py", title="💬 AI Chat Assistant", icon="💬")
st.navigation([page]).run()

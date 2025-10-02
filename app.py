import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

# Sidebar
st.sidebar.title("🔎 Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Arxiv Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Wikipedia Tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# DuckDuckGo Search Tool
search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 Multi-Tool Search Engine with Groq")

"""
In this example, we are using 'StreamlitCallbackHandler' to display the thoughts and actions of the agent in an interactive Streamlit App.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input(placeholder="Ask me anything about AI, ML, or LangChain!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API Key to continue.")
        st.stop()

    llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=api_key, streaming=True)
    tools = [wiki, arxiv, search]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(
            st.session_state.messages, callbacks=[st_callback]
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
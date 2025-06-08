"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Avratanu Biswas
# Updated by: ChatGPT
"""

import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# ✅ Custom prompt (replacement for deprecated ENTITY_MEMORY_CONVERSATION_TEMPLATE)
ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate.from_template(
    """
    You are a helpful assistant. Use the context and conversation history to answer the user's question.

    Context: {history}
    Current Input: {input}
    Answer:
    """
)

# Streamlit UI setup
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

# Session state initialization
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Input box
def get_text():
    return st.text_input("You: ", st.session_state["input"], key="input",
                         placeholder="Your AI assistant here! Ask me anything ...",
                         label_visibility='hidden')

# Reset chat
def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User: " + st.session_state["past"][i])
        save.append("Bot: " + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    if "entity_memory" in st.session_state:
        st.session_state.entity_memory.entity_store.clear()
        st.session_state.entity_memory.buffer.clear()

# Sidebar config
with st.sidebar.expander("🛠️ Options", expanded=False):
    if st.checkbox("Preview memory store"):
        with st.expander("Memory Store"):
            st.write(st.session_state.entity_memory.store if "entity_memory" in st.session_state else {})
    if st.checkbox("Preview memory buffer"):
        with st.expander("Buffer Store"):
            st.write(st.session_state.entity_memory.buffer if "entity_memory" in st.session_state else [])
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo'])
    K = st.number_input('(#) Summary of prompts to consider', min_value=3, max_value=1000, value=5)

st.title("🤖 Chat Bot with 🧠")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

API_O = st.sidebar.text_input("🔑 OPENAI-API-KEY", type="password")

# App logic only runs if API key is provided
if API_O:
    llm = ChatOpenAI(temperature=0, openai_api_key=API_O, model_name=MODEL)

    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
        verbose=False
    )
else:
    st.sidebar.warning('⚠️ Please enter your OpenAI API key to use the chatbot.')
    st.stop()

# New chat button
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Handle user input
user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display conversation
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    if download_str:
        st.download_button('Download Conversation', '\n'.join(download_str))

# Display stored chat sessions
for i, sublist in enumerate(st.session_state["stored_session"]):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Clear all chat history
if st.session_state["stored_session"]:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state["stored_session"]

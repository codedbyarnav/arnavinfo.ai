"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Avratanu Biswas
# Date: March 11, 2023
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your AI assistant here! Ask me anything ...",
                               label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    save = []
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append("User: " + st.session_state["past"][i])
        save.append("Bot: " + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    if "entity_memory" in st.session_state:
        st.session_state.entity_memory.entity_store.clear()
        st.session_state.entity_memory.buffer.clear()

# Sidebar
with st.sidebar.expander("🛠️", expanded=False):
    if st.checkbox("Preview memory store"):
        with st.expander("Memory Store"):
            st.write(st.session_state.get("entity_memory", {}).entity_store)
    if st.checkbox("Preview memory buffer"):
        with st.expander("Buffer Store"):
            st.write(st.session_state.get("entity_memory", {}).buffer)
    MODEL = st.selectbox('Model', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'])  # Use chat models only
    K = st.number_input(' (#)Summary of prompts to consider', min_value=3, max_value=1000)

# Main UI
st.title("🤖 Chat Bot with 🧠")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask for API Key
API_O = st.sidebar.text_input("OPENAI-API-KEY", type="password")

# Set up the conversation chain
if API_O:
    # Instantiate ChatOpenAI (new API)
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=API_O,
        model_name=MODEL,
        verbose=False
    )

    # Create memory if not exists
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

    # ConversationChain setup
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
        verbose=False
    )

else:
    st.sidebar.warning("API key required to try this app. The API key is not stored in any form.")

# Start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get user input
user_input = get_text()

# Generate and store response
if user_input and API_O:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display conversation history
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)

# Show previous sessions
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session: {i}"):
        st.write(sublist)

# Option to clear all
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear all"):
        del st.session_state.stored_session

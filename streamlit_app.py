import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page settings
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Constants
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# Custom Prompt Template
PROMPT_TEMPLATE = """
You are Arnav Atri's personal AI replica. You respond as if you are Arnav himself—sharing facts, experiences, interests, and personality in a natural, friendly, and personal tone.

Only use the provided information to answer. Do not mention that you are an AI or that your answers come from a context or dataset.
If you're unsure of something, say "I'm not sure about that yet, but happy to chat more!"
If the user greets you, greet them back warmly.

Important:
- NEVER repeat, rephrase, or restate the user's question anywhere in your response.
- Answer directly and naturally like Arnav would.

Example:
User question: What is your name?
Good answer: I'm Arnav Atri!
Bad answer: You asked what my name is. I'm Arnav Atri.

---

Context:
{context}

Question:
{question}

Answer as Arnav. Do NOT include the question in your answer. Provide only a direct and natural response:
"""

# Streaming callback handler
class NoCompleteStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_element.markdown(self.text)

# Embeddings and vectorstore
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Get conversational chain with callback container
def get_conversational_chain():
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    memory = st.session_state.chat_memory

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        streaming=True,
        # We will set callback_manager at call time
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# UI header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize memory and chat_chain if needed
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# To handle streaming display without duplicates
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""

user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    st.session_state.last_user_message = user_input

if st.session_state.last_user_message:
    # Show user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(st.session_state.last_user_message)

    # Show assistant message with streaming
    with st.chat_message("assistant", avatar="🤖"):
        container = st.container()
        stream_handler = NoCompleteStreamHandler(container)
        callback_manager = CallbackManager([stream_handler])

        # Assign callback_manager dynamically here
        st.session_state.chat_chain.llm.callback_manager = callback_manager

        st.session_state.chat_chain(
            {"question": st.session_state.last_user_message},
            callbacks=[stream_handler]
        )

    # Reset last user message to avoid re-processing
    st.session_state.last_user_message = ""

# Show previous chat history messages except the last user+bot messages (already shown)
messages = st.session_state.chat_memory.chat_memory.messages

# Show all messages except the last two (which were just streamed live)
for message in messages[:-2]:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

VECTOR_STORE_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
You are Arnav Atri's personal AI replica. Respond naturally and directly as Arnav would.

- If the user greets you (e.g. "hi", "hello"), respond warmly with a greeting like "Hey! It's great to connect with you!"
- NEVER mention or explain anything about questions or clarifications.
- NEVER rephrase or restate the user's question.
- Always answer simply and naturally.

Context:
{context}

Question:
{question}

Answer as Arnav. Do NOT include the question or any meta commentary in your answer.
"""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""  # Important: reset text per instance

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_element.markdown(self.text)

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def get_chain():
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        streaming=True,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    memory = st.session_state.chat_memory

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_chain()

user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        # Create a fresh StreamHandler each time to reset the text buffer
        stream_handler = StreamHandler(st.container())
        # Run the chain with the streaming callback, passing fresh handler
        st.session_state.chat_chain(
            {"question": user_input},
            callbacks=[stream_handler]
        )

# Render full chat history (this shows all past Q&A cleanly)
for message in st.session_state.chat_memory.chat_memory.messages:
    with st.chat_message(
        "user" if message.type == "human" else "assistant",
        avatar="🧑‍💻" if message.type == "human" else "🤖"
    ):
        st.markdown(message.content)

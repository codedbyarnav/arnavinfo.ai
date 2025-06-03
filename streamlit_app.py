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

VECTOR_STORE_PATH = "vectorstore/db_faiss"

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

class NoCompleteStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_element.markdown(self.text)

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def create_conversational_chain(callbacks=None):
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    memory = st.session_state.chat_memory

    callback_manager = CallbackManager(callbacks) if callbacks else CallbackManager([])

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        streaming=bool(callbacks),
        callback_manager=callback_manager
    )

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

user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    response_container = st.container()

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        stream_handler = NoCompleteStreamHandler(response_container)
        chat_chain = create_conversational_chain(callbacks=[stream_handler])

        response = chat_chain(
            {"question": user_input}
        )

    # Add to chat memory after response
    st.session_state.chat_memory.chat_memory.add_user_message(user_input)
    st.session_state.chat_memory.chat_memory.add_ai_message(response["answer"])

# Render chat history once
for message in st.session_state.chat_memory.chat_memory.messages:
    with st.chat_message(
        "user" if message.type == "human" else "assistant",
        avatar="🧑‍💻" if message.type == "human" else "🤖"
    ):
        st.markdown(message.content)

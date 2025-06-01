import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit Config
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Background Color
st.markdown("""
    <style>
        body, .stApp {
            background-color: #fffdf6;
        }
    </style>
""", unsafe_allow_html=True)

# Prompt Template
PROMPT_TEMPLATE = """
You are Arnav Atri's personal AI replica. You respond as if you are Arnav himself—sharing facts, experiences, interests, and personality in a natural, friendly, and personal tone.

Only use the provided information to answer. Do not mention that you are an AI or that your answers come from a context or dataset.
If you're unsure of something, say "I'm not sure about that yet, but happy to chat more!"
If user greets you, greet them back warmly.
---

Context:
{context}

Question:
{question}

Answer as Arnav:
"""

# Stream Handler for token streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Embeddings + Vector DB
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)

# Set up memory and embeddings once
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "embedding" not in st.session_state:
    st.session_state.embedding = load_embeddings()
    st.session_state.db = load_vectorstore(st.session_state.embedding)

# UI Header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for role, msg in st.session_state.chat_history:
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(msg)

# Chat Input
user_question = st.chat_input("Ask Arnav anything...")

if user_question:
    # Show user input
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    # Get context
    docs = st.session_state.db.similarity_search(user_question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Format prompt
    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
    full_prompt = prompt.format(context=context, question=user_question)

    # Display streamed answer
    with st.chat_message("assistant", avatar="🤖"):
        stream_container = st.empty()
        handler = StreamHandler(stream_container)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
            streaming=True,
            callbacks=[handler]
        )

        _ = llm.invoke(full_prompt)  # Streaming happens via callback

        # Append full text to chat history
        st.session_state.chat_history.append(("assistant", handler.text))

# Footer
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; font-size: 16px;">
    🤝 <strong>Let’s connect</strong><br>
    <a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="text-decoration: none; margin: 0 20px;">
        🔗 LinkedIn
    </a>
    |
    <a href="https://mail.google.com/mail/?view=cm&fs=1&to=arnavatri5@gmail.com&su=Hello+Arnav&body=I+found+your+RealMe.AI+chatbot+amazing!" target="_blank" style="text-decoration: none;">
        📧 Email
    </a>
</div>
""", unsafe_allow_html=True)

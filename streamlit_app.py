import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(page_title="RealMe.AI", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.header {
    background-color: #003366;
    padding: 30px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.header h1 {
    margin: 0;
    font-size: 2.5em;
}
.header p {
    font-size: 1.2em;
    margin-top: 10px;
}
.chat-bubble-user {
    background-color: #d1e3ff;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 85%;
}
.chat-bubble-bot {
    background-color: #e6f4f1;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 85%;
}
.footer {
    margin-top: 40px;
    text-align: center;
}
.footer a {
    background-color: #003366;
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    margin: 0 10px;
    text-decoration: none;
}
.footer a:hover {
    background-color: #005580;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>RealMe.AI</h1>
    <p>Ask anything about Arnav Atri — his projects, passions, and journey</p>
</div>
""", unsafe_allow_html=True)

# Prompt template
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

# Vector + chain setup
VECTOR_STORE_PATH = "vectorstore/db_faiss"

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Initialize session
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# Chat history display
for message in st.session_state.chat_chain.memory.chat_memory.messages:
    if message.type == "human":
        st.markdown(f"<div class='chat-bubble-user'><strong>You:</strong><br>{message.content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'><strong>Arnav:</strong><br>{message.content}</div>", unsafe_allow_html=True)

# Input
user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    st.markdown(f"<div class='chat-bubble-user'><strong>You:</strong><br>{user_input}</div>", unsafe_allow_html=True)

    response = st.session_state.chat_chain({"question": user_input})
    bot_reply = response["answer"]

    st.markdown(f"<div class='chat-bubble-bot'><strong>Arnav:</strong><br>{bot_reply}</div>", unsafe_allow_html=True)

# Footer with contact buttons
st.markdown("""
<div class="footer">
    <a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank">Connect on LinkedIn</a>
    <a href="https://mail.google.com/mail/?view=cm&fs=1&to=arnavatri5@gmail.com&su=Hello+Arnav&body=I+found+your+RealMe.AI+chatbot+amazing!" target="_blank">Email Arnav</a>
</div>
""", unsafe_allow_html=True)

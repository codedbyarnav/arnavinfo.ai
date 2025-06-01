import os
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="RealMe.AI", page_icon="🧠")

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_STORE_PATH = "vectorstore/db_faiss"

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

st.title("🧠 RealMe.AI - Ask Me Anything About Arnav")

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

for message in st.session_state.chat_chain.memory.chat_memory.messages:
    with st.chat_message("user" if message.type == "human" else "assistant"):
        st.markdown(message.content)

user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.chat_chain({"question": user_input})
    bot_reply = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="text-decoration: none; margin-right: 30px;">
            🔗 <strong>LinkedIn</strong>
        </a>
        <a href="https://mail.google.com/mail/?view=cm&fs=1&to=arnavatri5@gmail.com&su=Hello+Arnav&body=I+found+your+RealMe.AI+chatbot+amazing!" target="_blank">
            📧 <strong>Email</strong>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


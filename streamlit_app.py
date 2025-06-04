import os
import json
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler

# --- Custom Stream Handler for streaming ---
class NoCompleteStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_element.markdown(self.text)

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Page config ---
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are Arnav Atri's personal AI replica. You respond as if you are Arnav himself—sharing facts, experiences, interests, and personality in a natural, friendly, and personal tone.

Only use the provided information to answer. Do not mention that you are an AI or that your answers come from a context or dataset.
If you're unsure of something, say "I'm not sure about that yet, but happy to chat more!"
If the user greets you, greet them back warmly.

Important:
- NEVER repeat, rephrase, or restate the user's question anywhere in your response.
- Answer directly and naturally like Arnav would.

---

Context:
{context}

User: {question}

Arnav:
"""

# --- Memory JSON File ---
MEMORY_FILE = "memory.json"

def load_memory_json():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory_json(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def process_user_memory_input(user_input, memory):
    if user_input.lower().startswith("remember "):
        try:
            _, key_value = user_input.split("remember ", 1)
            key, value = key_value.split(" is ", 1)
            memory[key.strip().lower()] = value.strip()
            save_memory_json(memory)
            return f"Got it! I'll remember that {key.strip()} is {value.strip()}."
        except:
            return "Please use format: remember <thing> is <description>."
    if user_input.lower().startswith("what is"):
        key = user_input.lower().replace("what is", "").strip().rstrip("?")
        if key in memory:
            return memory[key]
    return None

# --- Helpers ---
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        streaming=True,
        api_key=GROQ_API_KEY,
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

# --- UI Header ---
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# --- Init chat chain ---
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# Load persistent memory
if "json_memory" not in st.session_state:
    st.session_state.json_memory = load_memory_json()

# --- Chat Input ---
user_input = st.chat_input("Ask Arnav anything...")

# --- Chat Processing ---
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    memory_response = process_user_memory_input(user_input, st.session_state.json_memory)
    if memory_response:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(memory_response)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            container = st.container()
            stream_handler = NoCompleteStreamHandler(container)
            st.session_state.chat_chain(
                {"question": user_input},
                callbacks=[stream_handler]
            )

# --- Show full chat history ---
messages = st.session_state.chat_chain.memory.chat_memory.messages
for message in messages:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

# --- Footer ---
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

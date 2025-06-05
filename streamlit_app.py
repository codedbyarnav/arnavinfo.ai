import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Load OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit page setup
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

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

# Streaming callback handler that updates inside the assistant's chat bubble container
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)  # Overwrite container content each token

def get_conversational_chain():
    # We don't pass container here; it'll be passed at invoke time via callback injection
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True
        # callbacks are attached later dynamically
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
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# Page header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize chain once
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# Display previous chat messages
for msg in st.session_state.chat_chain.memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# User input and streaming response
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    # Show user's message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Show assistant's message container first,

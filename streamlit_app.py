import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Load API key securely
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Constants
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

# Streaming handler to stream inside chat bubble
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)

# Embeddings and vector store loader
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Setup memory, embeddings, retriever, LLM, and chain once
if "chat_chain" not in st.session_state:
    # Core components
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)
    retriever = vector_db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    dummy_container = st.empty()
    stream_handler = StreamHandler(dummy_container)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler],
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    st.session_state.chat_chain = chat_chain
    st.session_state.llm = llm

# Header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Show previous messages
for msg in st.session_state.chat_chain.memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# Input box
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖") as assistant_container:
        stream_placeholder = st.empty()
        stream_handler = StreamHandler(stream_placeholder)

        # Update the existing LLM's callback for live stream
        st.session_state.llm.callbacks = [stream_handler]

        # Run chain (which internally uses the same LLM + memory)
        st.session_state.chat_chain.invoke({"question": user_input})

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

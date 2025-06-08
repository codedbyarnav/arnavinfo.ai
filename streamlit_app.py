import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationEntityMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Load API key securely
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="🧠 RealMe.AI", layout="wide")

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

# Load embedding model and vector store
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Stream handler to display live token stream
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Sidebar with controls
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("Configure your chat preferences.")
    model = st.selectbox("LLM Model", options=["gpt-3.5-turbo"], index=0)
    st.session_state.K = st.number_input("# of turns to remember", min_value=3, max_value=100, value=5)
    st.button("New Chat", on_click=lambda: [
        st.session_state.chat_chain.memory.clear(),
        st.session_state.chat_history.clear()
    ])

# Initialize memory and embedding-backed chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_chain" not in st.session_state:
    base_llm = ChatOpenAI(model_name=model, openai_api_key=OPENAI_API_KEY)
    memory = ConversationEntityMemory(llm=base_llm, memory_key="chat_history", return_messages=True, k=st.session_state.K)
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)
    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
    st.session_state.chat_chain = ConversationalRetrievalChain.from_llm(
        llm=base_llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# Header
st.title("🧠 RealMe.AI")
st.subheader("Ask anything about Arnav Atri")
st.divider()

# Chat input and output
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖") as container:
        stream_placeholder = st.empty()
        stream_handler = StreamHandler(stream_placeholder)
        llm_stream = ChatOpenAI(model_name=model, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[stream_handler])
        embeddings = load_embeddings()
        vector_db = load_vectorstore(embeddings)
        prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
        st.session_state.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_stream,
            retriever=vector_db.as_retriever(),
            memory=st.session_state.chat_chain.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
        st.session_state.chat_chain.invoke({"question": user_input})

# Show past chat history
with st.expander("Conversation History", expanded=True):
    for msg in st.session_state.chat_chain.memory.chat_memory.messages:
        if msg.type == "human":
            st.info(msg.content, icon="🤖")
        else:
            st.success(msg.content, icon="😊")

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

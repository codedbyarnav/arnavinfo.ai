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

# Load embeddings and vector store
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Stream handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Build the conversational chain
def get_conversational_chain(stream_handler, memory):
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# Header UI
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize memory and handler (only once)
if "memory" not in st.session_state:
    base_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    st.session_state.memory = ConversationEntityMemory(llm=base_llm, memory_key="chat_history", return_messages=True)

if "chat_chain" not in st.session_state:
    dummy_container = st.empty()
    handler = StreamHandler(dummy_container)
    st.session_state.chat_chain = get_conversational_chain(handler, st.session_state.memory)

# Display chat history
chat_memory = st.session_state.memory.chat_memory
for msg in chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# Input box
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖") as assistant_container:
        stream_placeholder = st.empty()
        handler = StreamHandler(stream_placeholder)

        # Rebuild chain with current handler
        st.session_state.chat_chain = get_conversational_chain(handler, st.session_state.memory)

        # Ask the question (only one input key: question)
        st.session_state.chat_chain.invoke({
            "question": user_input
        })

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

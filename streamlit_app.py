import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler

# -------------------- Stream handler for token-by-token streaming --------------------
class NoCompleteStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_element.markdown(self.text)


# -------------------- Load environment variables --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Streamlit page settings --------------------
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# -------------------- Constants --------------------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# -------------------- Custom prompt --------------------
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

# -------------------- Helpers --------------------
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3,
        streaming=True,
        api_key=GROQ_API_KEY,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    # Disable rephrasing of follow-ups
    dummy_question_generator = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template("{question}")
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        question_generator=dummy_question_generator
    )

# -------------------- UI Setup --------------------
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# -------------------- Show chat history --------------------
for message in st.session_state.chat_chain.memory.chat_memory.messages:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

# -------------------- Input handling --------------------
user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        stream_handler = NoCompleteStreamHandler(st.container())
        st.session_state.chat_chain(
            {"question": user_input},
            callbacks=[stream_handler]
        )

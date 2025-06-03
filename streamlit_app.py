import streamlit as st
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
import pickle

# Custom Streamlit callback handler for streaming tokens
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # Show typing cursor

    def on_llm_end(self, *args, **kwargs) -> None:
        self.container.markdown(self.text)  # Final output

# Load your vectorstore
with open("vectorstore/index.pkl", "rb") as f:
    vectorstore = pickle.load(f)


# Initialize embeddings (match your vectorstore embeddings)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Groq chat model with streaming enabled
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama3-8b-8192",
    streaming=True,
)

# Setup conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# Streamlit UI
st.set_page_config(page_title="CampusAI", page_icon="🎓")
st.title("🎓 CampusAI - Ask about your college!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything about your college...")

async def run_chain(question):
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    result = await chain.acall({"question": question}, callbacks=[stream_handler])
    return result["answer"]

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the chain asynchronously and get the answer with streaming
    answer = asyncio.run(run_chain(user_input))

    # Save conversation
    st.session_state.chat_history.append((user_input, answer))

# Display full chat history
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

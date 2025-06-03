# streamlit_app.py

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StreamHandler

# Load FAISS index
db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Setup prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the following context. If you don't know, say you don't know. Don't make up anything.

    Context:
    {context}

    Question:
    {question}
    """
)

# Groq LLM setup with streaming
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # or llama3-8b
    temperature=0.2,
    streaming=True
)

# Chain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chain for RAG + Chat
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Streamlit UI
st.title("Campus AI Chatbot 🤖")
st.markdown("Ask questions based on DSEU information.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = chain.invoke(
            {"question": user_input},
            callbacks=[stream_handler]
        )
        st.session_state.chat_history.append((user_input, response["answer"]))

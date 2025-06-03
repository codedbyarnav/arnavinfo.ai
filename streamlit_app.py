import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatGroq
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents import StuffDocumentsChain

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT make up an answer.

{context}

Question: {question}

Helpful Answer:
"""

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain(vector_db):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3,
        streaming=False,  # ❗ must be False with LangChain chains
        api_key=GROQ_API_KEY,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    return ConversationalRetrievalChain(
        retriever=vector_db.as_retriever(),
        combine_docs_chain=stuff_chain,
        memory=memory,
    )

def main():
    st.set_page_config(page_title="CampusAI Chatbot", page_icon="🤖")
    st.header("Ask about DSEU 📚")

    user_input = st.text_input("Ask a question about your college:")

    if user_input:
        vector_db = load_vector_store()
        if "chat_chain" not in st.session_state:
            st.session_state.chat_chain = get_conversational_chain(vector_db)

        response = st.session_state.chat_chain(
            {"question": user_input}
        )

        st.write(response["answer"])

if __name__ == "__main__":
    main()

# embed_pdfs.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load PDF files
def load_pdffiles(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
    #return documnes.faiss

# Step 2: Split documents into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

# Step 3: Initialize HF embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create and save FAISS vector store
def create_vector_store(chunks, embedding_model, db_path):
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(db_path)

if __name__ == "__main__":
    documents = load_pdffiles(DATA_PATH)
    chunks = create_chunks(documents)
    embeddings = get_embedding_model()
    create_vector_store(chunks, embeddings, DB_FAISS_PATH)
    print("✅ Embeddings created and stored in FAISS.")

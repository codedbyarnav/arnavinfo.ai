import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Comprehensive response cleaner
def clean_response(response: str, original_question: str = "") -> str:
    """Remove question repetition and reformulation from response"""
    
    # Remove common question reformulation patterns
    patterns_to_remove = [
        "Here is the rephrased standalone question:",
        "Here's the rephrased question:",
        "Rephrased question:",
        "The question is:",
        "Question:",
        "Here is the answer:",
        "Answer:",
    ]
    
    for pattern in patterns_to_remove:
        if pattern in response:
            response = response.split(pattern)[-1]
    
    # Split by lines and find where the actual answer starts
    lines = response.split('\n')
    cleaned_lines = []
    answer_started = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if answer_started:
                cleaned_lines.append("")
            continue
        
        # Skip lines that look like question repetition
        if (line.lower().startswith(('tell me', 'what is', 'what are', 'how', 'when', 'where', 'why', 'can you')) 
            and not answer_started):
            continue
            
        # Look for the start of actual answer
        if (line.lower().startswith(('i\'m arnav', 'my name is', 'i am arnav', 'hello', 'hi there', 'i\'m ', 'i am ')) 
            or 'arnav atri' in line.lower() 
            or any(phrase in line.lower() for phrase in ['i really', 'i love', 'i work', 'i study', 'i enjoy'])):
            answer_started = True
            cleaned_lines.append(line)
        elif answer_started:
            cleaned_lines.append(line)
        elif not any(char in line for char in '?'):  # If no question mark, might be answer
            answer_started = True
            cleaned_lines.append(line)
    
    # Join the cleaned lines
    result = '\n'.join(cleaned_lines).strip()
    
    # If we didn't find a clear answer start, return everything after first few lines
    if not result or len(result) < 20:
        lines = response.split('\n')
        # Skip first 1-2 lines if they look like questions
        start_idx = 0
        for i, line in enumerate(lines[:3]):
            if line.strip() and ('?' in line or any(q in line.lower() for q in ['tell me', 'what is', 'what are'])):
                start_idx = i + 1
            else:
                break
        result = '\n'.join(lines[start_idx:]).strip()
    
    return result if result else response

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page settings
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Constants
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# Improved Custom Prompt with stronger instructions
PROMPT_TEMPLATE = """
You are Arnav Atri speaking directly. Answer naturally as yourself.

STRICT RULES:
1. DO NOT repeat the user's question
2. DO NOT say "Here is the rephrased question" or similar
3. DO NOT reformulate or rephrase the question
4. Start immediately with your answer as Arnav
5. Be conversational and personal

Context: {context}
User: {question}

Arnav:""".strip()

# Load Hugging Face embeddings
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vectorstore
def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Set up LangChain conversation chain
def get_conversational_chain():
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0.3,
        streaming=False,  # Disable streaming for clean post-processing
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

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Load or create the chain
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_conversational_chain()

# Display chat history
for message in st.session_state.chat_chain.memory.chat_memory.messages:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

# Chat input box
user_input = st.chat_input("Ask Arnav anything...")

if user_input:
    # Show user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Show assistant message
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            # Get response without streaming
            response = st.session_state.chat_chain({"question": user_input})
            
            # Clean the response
            cleaned_response = clean_response(response["answer"], user_input)
            
            # Display cleaned response
            st.markdown(cleaned_response)

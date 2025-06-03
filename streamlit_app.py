import os
from dotenv import load_dotenv
import streamlit as st
import re

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage

# Custom Streamlit callback handler for clean streaming
class CleanStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.text = ""
        self.full_response = ""
        self.started_answer = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_response += token
        
        # Look for patterns that indicate we've moved past question reformulation
        if not self.started_answer:
            # Check if we've hit the actual answer part
            if any(indicator in self.full_response.lower() for indicator in [
                "i'm arnav", "my name is", "i am arnav", "hello there", "hi there",
                "i really", "i enjoy", "i love", "i work", "i study"
            ]):
                # Find where the answer actually starts
                lines = self.full_response.split('\n')
                answer_lines = []
                found_start = False
                
                for line in lines:
                    if found_start:
                        answer_lines.append(line)
                    elif any(indicator in line.lower() for indicator in [
                        "i'm arnav", "my name is", "i am arnav", "hello there", "hi there",
                        "i really", "i enjoy", "i love", "i work", "i study"
                    ]):
                        found_start = True
                        answer_lines.append(line)
                
                self.text = '\n'.join(answer_lines)
                self.started_answer = True
        else:
            self.text += token
        
        # Only display if we've started the actual answer
        if self.started_answer:
            self.text_element.markdown(self.text)

# Function to extract context from vector store
def get_relevant_context(question: str, vector_db, k=4):
    docs = vector_db.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

# Function to clean response from any remaining artifacts
def final_clean_response(response: str) -> str:
    # Remove any remaining question reformulation patterns
    patterns = [
        r"Here is the rephrased standalone question:.*?\?",
        r"Rephrased question:.*?\?",
        r"The question is:.*?\?",
        r"What do you do and what are you interested in\?",
        r"Tell me about yourself\?",
        r".*?standalone question.*?\?"
    ]
    
    for pattern in patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up any remaining artifacts
    lines = response.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not any(unwanted in line.lower() for unwanted in [
            'here is the rephrased', 'rephrased question', 'what do you do and what are you'
        ]):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page settings
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Constants
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# Direct prompt template without conversation chain complexity
DIRECT_PROMPT_TEMPLATE = """
You are Arnav Atri. Respond naturally and directly as yourself. Use the provided context to answer accurately.

CRITICAL: 
- Do NOT repeat the user's question
- Do NOT say "Here is the rephrased question" or similar phrases
- Start immediately with your natural response as Arnav
- Be conversational and personal

Context about Arnav:
{context}

User asked: {question}

Arnav responds:"""

# Load Hugging Face embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0.3,
        streaming=True,
        api_key=GROQ_API_KEY,
    )

# Custom conversation function
def get_response(question: str, chat_history: list):
    vector_db = load_vectorstore()
    llm = get_llm()
    
    # Get relevant context
    context = get_relevant_context(question, vector_db)
    
    # Create prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=DIRECT_PROMPT_TEMPLATE
    )
    
    # Format the prompt
    formatted_prompt = prompt.format(context=context, question=question)
    
    return llm, formatted_prompt

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask Arnav anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    
    # Get and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        try:
            llm, formatted_prompt = get_response(user_input, st.session_state.messages)
            
            # Stream the response
            response_container = st.empty()
            full_response = ""
            
            # Stream tokens one by one
            for chunk in llm.stream(formatted_prompt):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                else:
                    token = str(chunk)
                
                full_response += token
                
                # Clean the response in real-time
                cleaned_response = final_clean_response(full_response)
                response_container.markdown(cleaned_response)
            
            # Final cleanup and add to chat history
            final_response = final_clean_response(full_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})

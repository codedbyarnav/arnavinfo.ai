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
from langchain.schema import BaseMessage

# -------------------- Stream handler with response cleaning --------------------
class CleanStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text_element = container.empty()
        self.full_text = ""
        self.cleaned_text = ""
        self.answer_started = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_text += token
        
        # Clean the response in real-time
        cleaned = self.clean_response_realtime(self.full_text)
        if cleaned != self.cleaned_text:
            self.cleaned_text = cleaned
            self.text_element.markdown(self.cleaned_text)

    def clean_response_realtime(self, text: str) -> str:
        # Remove question reformulation patterns
        patterns_to_remove = [
            r"Here is the rephrased standalone question:.*?\n",
            r"Rephrased question:.*?\n",
            r"The question is:.*?\n",
            r".*?standalone question.*?\n",
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Split by lines and find actual answer start
        lines = text.split('\n')
        cleaned_lines = []
        found_answer = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if found_answer:
                    cleaned_lines.append("")
                continue
                
            # Skip obvious question repetitions
            if (line.lower().startswith(('tell me', 'what is', 'what are', 'how', 'when', 'where', 'why'))
                and '?' in line and not found_answer):
                continue
                
            # Look for answer indicators
            if (any(indicator in line.lower() for indicator in [
                "i'm arnav", "my name is", "i am arnav", "hello", "hi there",
                "i really", "i enjoy", "i love", "i work", "i study", "i'm passionate"
            ]) or found_answer):
                found_answer = True
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

# -------------------- Load environment variables --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Streamlit page settings --------------------
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# -------------------- Constants --------------------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

# -------------------- Enhanced prompt --------------------
PROMPT_TEMPLATE = """
You are Arnav Atri speaking directly to someone. Answer naturally as yourself using the provided context.

CRITICAL RULES:
- Do NOT repeat the user's question
- Do NOT say "Here is the rephrased question" or similar
- Start immediately with your natural response as Arnav
- Be conversational and personal
- Use "I" statements naturally

Context about you:
{context}

User: {question}

Arnav: """

# -------------------- Custom RAG implementation --------------------
class CustomRAGChain:
    def __init__(self, llm, retriever, memory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE
        )
    
    def __call__(self, inputs, callbacks=None):
        question = inputs["question"]
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        # Get response from LLM
        response = ""
        if callbacks:
            for chunk in self.llm.stream(formatted_prompt, callbacks=callbacks):
                if hasattr(chunk, 'content'):
                    response += chunk.content
        else:
            response = self.llm.invoke(formatted_prompt).content
        
        # Add to memory
        from langchain.schema import HumanMessage, AIMessage
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)
        
        return {"answer": response}

# -------------------- Helpers --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def get_chat_chain():
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0.3,
        streaming=True,
        api_key=GROQ_API_KEY,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    vector_db = load_vectorstore()
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    return CustomRAGChain(llm, retriever, memory)

# -------------------- UI Setup --------------------
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize chat chain
if "chat_chain" not in st.session_state:
    try:
        st.session_state.chat_chain = get_chat_chain()
    except Exception as e:
        st.error(f"Error initializing chat chain: {str(e)}")
        st.stop()

# -------------------- Show chat history --------------------
if hasattr(st.session_state.chat_chain, 'memory') and st.session_state.chat_chain.memory.chat_memory.messages:
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
        try:
            stream_handler = CleanStreamHandler(st.container())
            response = st.session_state.chat_chain(
                {"question": user_input},
                callbacks=[stream_handler]
            )
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Fallback: try without streaming
            try:
                response = st.session_state.chat_chain({"question": user_input})
                st.markdown(response.get("answer", "Sorry, I couldn't generate a response."))
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")

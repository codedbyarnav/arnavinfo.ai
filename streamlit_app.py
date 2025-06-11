import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.callbacks.base import BaseCallbackHandler

# Load API key securely
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Prompt Template
PROMPT_TEMPLATE = """
You are Arnav Atri's AI twin. You will carry a memory of Arnav's life and conversations with users.

Maintain friendly tone, respond with Arnav's perspective, and use remembered facts about people, places, or preferences as the chat continues.

Current conversation history:
{history}

Entities so far:
{entities}

New user input:
{input}

Reply as Arnav:
"""

# Streaming handler to stream inside chat bubble
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Chain builder with ConversationalEntityMemory
def get_entity_memory_chain(stream_handler):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler],
    )

    memory = ConversationEntityMemory(llm=llm)

    prompt = PromptTemplate(
        input_variables=["history", "input", "entities"],
        template=PROMPT_TEMPLATE,
    )

    return LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# Header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize chat chain once
if "chat_chain" not in st.session_state:
    dummy_container = st.empty()
    stream_handler = StreamHandler(dummy_container)
    st.session_state.chat_chain = get_entity_memory_chain(stream_handler)

# Show previous messages (newest at bottom)
for message in st.session_state.chat_chain.memory.chat_memory.messages:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

# Input box
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖") as assistant_container:
        stream_placeholder = st.empty()
        stream_handler = StreamHandler(stream_placeholder)

        # Create new LLMChain with streaming and memory
        chat_chain = get_entity_memory_chain(stream_handler)
        st.session_state.chat_chain = chat_chain  # Replace to retain memory

        # Ask the question
        chat_chain.invoke({"input": user_input})

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

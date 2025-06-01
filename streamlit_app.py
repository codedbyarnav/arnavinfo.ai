import os
from dotenv import load_dotenv
import streamlit as st

# Import Gemini SDK - adjust import if your SDK differs
from google.ai import generativeai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini SDK
generativeai.configure(api_key=GOOGLE_API_KEY)

# Constants and prompt
PROMPT_TEMPLATE = """
You are Arnav Atri's personal AI replica. You respond as if you are Arnav himself—sharing facts, experiences, interests, and personality in a natural, friendly, and personal tone.

Only use the provided information to answer. Do not mention that you are an AI or that your answers come from a context or dataset.
If you're unsure of something, say "I'm not sure about that yet, but happy to chat more!"
If user greets you, greet them back warmly.
---

Question:
{question}

Answer as Arnav:
"""

def stream_gemini_response(question: str):
    prompt = PROMPT_TEMPLATE.format(question=question)
    response_stream = generativeai.chat.completions.create(
        model="gemini-1.5-chat-bison",
        prompt=prompt,
        temperature=0.3,
        stream=True
    )
    for chunk in response_stream:
        # Assuming chunk has a text attribute with partial response
        yield chunk.text

def main():
    st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

    st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask Arnav anything...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Placeholder for assistant response
        message_placeholder = st.chat_message("assistant", avatar="🤖")
        full_response = ""

        # Stream the response and update UI
        for token in stream_gemini_response(user_input):
            full_response += token
            message_placeholder.markdown(full_response + "▌")  # Show cursor while streaming

        # Replace cursor with final response
        message_placeholder.markdown(full_response)

        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Footer with contact links
    st.markdown("""
    <hr style="margin-top: 30px;">
    <div style="text-align: center; font-size: 16px;">
    🤝 <strong>Let’s connect</strong><br>
    <a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="text-decoration: none; margin: 0 20px;">
    🔗 LinkedIn
    </a>
    |
    <a href="mailto:arnavatri5@gmail.com" target="_blank" style="text-decoration: none;">
    📧 Email
    </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

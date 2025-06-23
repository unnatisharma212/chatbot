import streamlit as st
from hcl_chatbot import HCLChatbot  # Import your chatbot class

# Set Streamlit page configuration FIRST
st.set_page_config(page_title="HCL Internal Assistant", page_icon="🤖")

# Fetch API key from Streamlit secrets
api_key = st.secrets["COHERE_API_KEY"]

# --- Title and Description ---
st.title("🤖 HCL Internal Assistant")
st.markdown("Ask me anything about HCL internal documents or general queries. I can use both internal knowledge and web search.")

# --- Initialize Chatbot (once per session) ---
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = HCLChatbot(
            data_folder="data",
            collection_name="hcl_docs_web_v2",
            cohere_api_key=api_key
        )
        st.success("Chatbot initialized successfully! Ready to chat.")
    except ValueError as e:
        st.error(f"Initialization Error: {e}. Please ensure COHERE_API_KEY is set in Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during chatbot initialization: {e}")
        st.stop()
else:
    st.info("Chatbot is ready!")

# Access the chatbot instance
chatbot = st.session_state.chatbot

# --- Chat History Display ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chat Logic ---
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            local_docs = []
            if chatbot.collection:
                local_docs = chatbot.retrieve_local_documents(prompt)
            else:
                st.info("No local documents loaded or ChromaDB not configured for local retrieval.")

            response = chatbot.generate_final_response(prompt, local_docs)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Optional: Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


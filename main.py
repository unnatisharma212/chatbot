import streamlit as st
from hcl_chatbot import HCLChatbot  # Import your chatbot class
st.write("Loaded API key:", st.secrets.get("COHERE_API_KEY", "NOT FOUND"))

# Fetch API key from Streamlit secrets
api_key = st.secrets["COHERE_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title="HCL Internal Assistant", page_icon="🤖")

# --- Title and Description ---
st.title("🤖 HCL Internal Assistant")
st.markdown("Ask me anything about HCL internal documents or general queries. I can use both internal knowledge and web search.")

# --- Initialize Chatbot (once per session) ---
# Use st.session_state to persist the chatbot instance across reruns
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = HCLChatbot(
            data_folder="data",
            collection_name="hcl_docs_web_v2",
            cohere_api_key=api_key  # Pass the key to the chatbot
        )
        st.success("Chatbot initialized successfully! Ready to chat.")
    except ValueError as e:
        st.error(f"Initialization Error: {e}. Please ensure COHERE_API_KEY is set in Streamlit secrets.")
        st.stop()  # Stop execution if key is missing
    except Exception as e:
        st.error(f"An unexpected error occurred during chatbot initialization: {e}")
        st.stop()
else:
    st.info("Chatbot is ready!")  # Indicate that chatbot is already initialized

# Access the chatbot instance
chatbot = st.session_state.chatbot

# --- Chat History Display ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chat Logic ---
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve relevant local documents
            local_docs = []
            if chatbot.collection:  # Check if local document retrieval is possible
                local_docs = chatbot.retrieve_local_documents(prompt)
            else:
                st.info("No local documents loaded or ChromaDB not configured for local retrieval.")

            # 2. Generate the response using local docs and allowing Cohere to use web search
            response = chatbot.generate_final_response(prompt, local_docs)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()  # Rerun the app to clear displayed messages

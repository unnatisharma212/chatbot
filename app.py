import os
import chromadb
import cohere
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HCLChatbot:
    """
    A chatbot for HCL using ChromaDB for semantic search of local documents
    and Cohere for generation, with web search capabilities.
    """
    def __init__(self, data_folder: str = "data", db_path: str = "chroma_db", collection_name: str = "hcl_docs_web"):
        """
        Initializes the chatbot.

        Args:
            data_folder (str): The name of the folder containing .txt documents.
            db_path (str): The path to store the ChromaDB persistent database.
            collection_name (str): The name of the ChromaDB collection.
        """
        self.script_dir = Path(__file__).parent
        self.data_folder = self.script_dir / data_folder
        self.db_path = str(self.script_dir / db_path) # Ensure path is a string
        self.collection_name = collection_name
        self.documents = [] # Stores loaded documents from the data_folder

        # Load Cohere API key and initialize client
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            logging.error("COHERE_API_KEY environment variable is not set.")
            raise ValueError("COHERE_API_KEY environment variable is not set. Please set it before running.")
        self.cohere_client = cohere.Client(self.cohere_api_key)

        # Load data and initialize ChromaDB
        self._load_data()
        self._initialize_chroma()

    def _load_data(self) -> None:
        """
        Loads text documents from the specified data folder.
        """
        logging.info(f"Looking for data in: {self.data_folder}")
        if not self.data_folder.exists():
            logging.warning(f"Data folder not found: {self.data_folder}. No local documents will be loaded.")
            return # Continue without local docs if folder doesn't exist

        loaded_count = 0
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(".txt"):
                file_path = self.data_folder / file_name
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip(): # Basic check for non-empty content
                            self.documents.append({"content": content, "id": file_name})
                            loaded_count += 1
                        else:
                            logging.warning(f"Skipping empty file: {file_name}")
                except Exception as e:
                    logging.error(f"Error reading file {file_name}: {e}")
        
        if loaded_count == 0 and self.data_folder.exists():
             logging.warning(f"No .txt documents were successfully loaded from {self.data_folder}.")
        elif loaded_count > 0 :
            logging.info(f"Loaded {loaded_count} text files from {self.data_folder}")


    def _initialize_chroma(self) -> None:
        """
        Initializes the ChromaDB client, collection, and adds documents if needed.
        This is only relevant if local documents are loaded.
        """
        if not self.documents:
            logging.info("No local documents loaded, skipping ChromaDB initialization for local document indexing.")
            self.collection = None # Explicitly set collection to None if no local docs
            return

        logging.info("Initializing ChromaDB for local document indexing...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func,
                metadata={"hnsw:space": "cosine"}
            )

            current_ids_in_db = self.collection.get(include=[])['ids'] # More efficient way to get IDs
            
            # Determine which documents are new
            new_documents_to_add = []
            new_metadatas_to_add = []
            new_ids_to_add = []
            
            existing_doc_ids_in_db = set(current_ids_in_db)

            for i, doc in enumerate(self.documents):
                # Create a unique ID for each document chunk if not already present
                # This example assumes doc['id'] is the filename and is unique enough for this purpose.
                # For more complex scenarios, consider content hashing or more robust ID generation.
                doc_id_to_check = f"{i}_{doc['id']}" 
                if doc_id_to_check not in existing_doc_ids_in_db:
                    new_documents_to_add.append(doc["content"])
                    new_metadatas_to_add.append({"source_type": "local_document", "source_name": doc["id"]})
                    new_ids_to_add.append(doc_id_to_check)

            if new_ids_to_add:
                logging.info(f"Adding {len(new_ids_to_add)} new documents to ChromaDB...")
                self.collection.add(
                    documents=new_documents_to_add,
                    metadatas=new_metadatas_to_add,
                    ids=new_ids_to_add
                )
                logging.info(f"Successfully added {len(new_ids_to_add)} new documents to ChromaDB.")
            elif self.documents:
                logging.info("ChromaDB collection is up-to-date with local documents.")
            
        except Exception as e:
            logging.error(f"Error initializing or updating ChromaDB: {e}")
            self.collection = None # Ensure collection is None if setup fails
            # Depending on the severity, you might want to raise the exception
            # raise

    def retrieve_local_documents(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieves relevant document chunks from ChromaDB based on the query.

        Args:
            query (str): The user's question.
            k (int): The number of documents to retrieve.

        Returns:
            List[Dict]: A list of retrieved documents with content and metadata.
                         Returns empty list if ChromaDB is not initialized or no docs found.
        """
        if not self.collection:
            logging.info("ChromaDB collection not available. Skipping local document retrieval.")
            return []

        logging.info(f"Retrieving local documents for query: '{query}'")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_docs = []
            if results and results.get("documents") and results["documents"][0]:
                 for i, doc_content in enumerate(results["documents"][0]):
                    retrieved_docs.append({
                        "content": doc_content,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
                 logging.info(f"Retrieved {len(retrieved_docs)} local documents.")
            else:
                logging.warning("No local documents found for the query in ChromaDB.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error retrieving documents from ChromaDB: {e}")
            return []

    def generate_with_cohere(self, query: str, prompt: str, documents: Optional[List[Dict]] = None) -> str:
        """
        Generates a response using the Cohere chat API, potentially with web search.

        Args:
            query (str): The original user query (can be used by Cohere for better web search).
            prompt (str): The formatted prompt for the Cohere model.
            documents (Optional[List[Dict]]): Documents from Cohere's RAG (e.g., web search results).
                                            This is for handling documents returned by Cohere's connectors.

        Returns:
            str: The generated response or an error message.
        """
        logging.info("Generating response with Cohere (web search enabled)...")
        try:
            # Using Cohere's chat API with web search connector
            # The 'documents' parameter in cohere.chat is for providing documents *to* the chat call,
            # which is useful if you've pre-fetched them.
            # If you want Cohere to perform the search, you use the `connectors` param.
            # The `prompt` already contains local context.
            
            chat_response = self.cohere_client.chat(
                message=prompt, # The prompt now contains the query and local context
                model="command-r-plus",
                temperature=0.2,
                connectors=[{"id": "web-search"}], # Enable web search
                # The 'documents' parameter here is if YOU provide documents to Cohere.
                # The web search connector will fetch its own documents.
                # We are passing our local context within the prompt.
            )
            logging.info("Cohere response received.")

            # The response object might contain citations or search results
            # that you can choose to display or process.
            # For simplicity, we're returning the main text response here.
            # You might want to inspect chat_response.documents or chat_response.citations
            # if you need to show where web information came from.
            
            response_text = chat_response.text
            
            # Optionally, add information about web documents used if available
            if hasattr(chat_response, 'documents') and chat_response.documents:
                web_doc_snippets = [f"Web Source: {doc.get('title', doc.get('url', 'Unknown'))[:50]}..." for doc in chat_response.documents if doc.get('id','').startswith('web-search')]
                if web_doc_snippets:
                    response_text += "\n\n(Information potentially supplemented by web search. Sources: " + "; ".join(web_doc_snippets) + ")"

            return response_text

        except Exception as e:
            logging.error(f"Error generating response with Cohere: {str(e)}")
            return f"Sorry, I encountered an error while trying to reach Cohere or process its response: {str(e)}"

    def generate_final_response(self, query: str, local_context_docs: List[Dict]) -> str:
        """
        Constructs the prompt using local context and then calls Cohere for generation,
        allowing Cohere to use web search as needed.

        Args:
            query (str): The user's original question.
            local_context_docs (List[Dict]): The list of documents retrieved from local ChromaDB.

        Returns:
            str: The chatbot's final answer.
        """
        context_str = "No local HCL documents found relevant to your query."
        if local_context_docs:
            context_str = "Context from HCL internal documents:\n"
            context_str += "\n\n".join([
                f"Source: {doc['metadata'].get('source_name', 'Unknown local file')}\nContent: {doc['content']}"
                for doc in local_context_docs
            ])

        # The prompt now instructs Cohere to use local context first, then web search if needed.
        prompt = f"""
You are an HCL Internal Assistant. Your primary goal is to answer the user's question based on the provided 'Context from HCL internal documents'.
If the answer is not found in the internal documents or if the internal documents are insufficient, you are permitted to use your web search capabilities to find the answer.

Instructions:
1.  Analyze the 'User Question' and the 'Context from HCL internal documents' provided below.
2.  **Prioritize information from 'Context from HCL internal documents'.** If a satisfactory answer is found there, synthesize a concise and professional answer based *only* on that information. Clearly state if the information comes from internal documents.
3.  If the internal documents do not provide an answer, or if the information is incomplete, use your web search ability to find the information.
4.  When using web search, try to cite the source or indicate that the information came from the web.
5.  If, after checking internal documents and performing a web search, you still cannot find a relevant answer, state: "I could not find sufficient information from internal documents or the web to answer your question."
6.  Do *not* make up information. Maintain a professional and helpful business tone.

---
{context_str}
---
User Question: {query}
---

Assistant Answer:
"""
        # The `generate_with_cohere` method will handle the actual call with web search enabled.
        # We pass the original query to Cohere as well, as it can sometimes improve its own search.
        return self.generate_with_cohere(query=query, prompt=prompt)


    def chat_interface(self):
        """
        Provides a command-line interface for interacting with the chatbot.
        """
        print("\n==============================================")
        print(" HCL Internal Assistant (AI-Powered + Web Search)")
        print("==============================================")
        print("Ask me questions. I can use HCL internal documents and the web.")
        print("Type 'exit' or 'quit' to end the conversation.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit']:
                    print("\nAssistant: Goodbye!")
                    break

                # 1. Retrieve relevant local documents (if ChromaDB is available)
                local_docs = []
                if self.collection: # Check if local document retrieval is possible
                    local_docs = self.retrieve_local_documents(user_input)
                else:
                    logging.info("Skipping local document retrieval as ChromaDB is not configured.")

                # 2. Generate the response using local docs and allowing Cohere to use web search
                response = self.generate_final_response(user_input, local_docs)

                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nAssistant: Session ended. Goodbye!")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred in the chat loop: {e}", exc_info=True)
                print("\nAssistant: Sorry, an unexpected error occurred. Please try again.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # You can specify a different data folder if needed.
        # If 'data' folder doesn't exist or is empty, it will rely more on web search.
        chatbot = HCLChatbot(data_folder="data", collection_name="hcl_docs_web_v2")
        chatbot.chat_interface()
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except RuntimeError as re:
        print(f"Runtime Error: {re}")
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        logging.error("Fatal error during chatbot initialization.", exc_info=True)





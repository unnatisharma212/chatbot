import sentence_transformers
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

    def __init__(self, data_folder: str = "data", db_path: str = "chroma_db", collection_name: str = "hcl_docs_web", cohere_api_key: Optional[str] = None):
        self.data_folder = Path(data_folder)
        self.db_path = db_path
        self.collection_name = collection_name
        self.documents = []

        # Use the argument if provided; fallback to environment variable
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not provided via argument or environment.")

        self.cohere_client = cohere.Client(self.cohere_api_key)

        # Load data and initialize ChromaDB
        self._load_data()
        self._initialize_chroma()

    def _load_data(self) -> None:
        logging.info(f"Looking for data in: {self.data_folder}")
        if not self.data_folder.exists():
            logging.warning(f"Data folder not found: {self.data_folder}. No local documents will be loaded.")
            return

        loaded_count = 0
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(".txt"):
                file_path = self.data_folder / file_name
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            self.documents.append({"content": content, "id": file_name})
                            loaded_count += 1
                        else:
                            logging.warning(f"Skipping empty file: {file_name}")
                except Exception as e:
                    logging.error(f"Error reading file {file_name}: {e}")

        if loaded_count == 0:
            logging.warning(f"No .txt documents were successfully loaded from {self.data_folder}.")
        else:
            logging.info(f"Loaded {loaded_count} text files from {self.data_folder}")

    def _initialize_chroma(self) -> None:
        if not self.documents:
            logging.info("No local documents loaded, skipping ChromaDB initialization.")
            self.collection = None
            return

        logging.info("Initializing ChromaDB...")
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

            current_ids_in_db = self.collection.get(include=[])['ids']
            existing_doc_ids_in_db = set(current_ids_in_db)

            new_documents_to_add = []
            new_metadatas_to_add = []
            new_ids_to_add = []

            for i, doc in enumerate(self.documents):
                doc_id = f"{i}_{doc['id']}"
                if doc_id not in existing_doc_ids_in_db:
                    new_documents_to_add.append(doc["content"])
                    new_metadatas_to_add.append({"source_type": "local_document", "source_name": doc["id"]})
                    new_ids_to_add.append(doc_id)

            if new_ids_to_add:
                logging.info(f"Adding {len(new_ids_to_add)} new documents to ChromaDB...")
                self.collection.add(
                    documents=new_documents_to_add,
                    metadatas=new_metadatas_to_add,
                    ids=new_ids_to_add
                )
                logging.info(f"Successfully added {len(new_ids_to_add)} documents to ChromaDB.")
            else:
                logging.info("ChromaDB collection is already up-to-date.")

        except Exception as e:
            logging.error(f"Error initializing ChromaDB: {e}")
            self.collection = None

    def retrieve_local_documents(self, query: str, k: int = 3) -> List[Dict]:
        if not self.collection:
            logging.info("No ChromaDB collection available.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            retrieved_docs = []
            if results.get("documents") and results["documents"][0]:
                for i, doc_content in enumerate(results["documents"][0]):
                    retrieved_docs.append({
                        "content": doc_content,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
                logging.info(f"Retrieved {len(retrieved_docs)} documents.")
            else:
                logging.warning("No matching documents found.")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error retrieving documents from ChromaDB: {e}")
            return []

    def generate_with_cohere(self, query: str, prompt: str, documents: Optional[List[str]] = None) -> str:
        try:
            response = self.cohere_client.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
                stop_sequences=["--END--"]
            )
            return response.generations[0].text.strip()
        except Exception as e:
            logging.error(f"Error generating response with Cohere: {str(e)}")
            return f"Sorry, an error occurred while generating a response: {str(e)}"

    def generate_final_response(self, user_query: str, local_docs: List[Dict]) -> str:
        try:
            context_string = "\n\n".join([doc["content"] for doc in local_docs])
            prompt = (
                f"Use the following HCL internal context and web knowledge if needed to answer the question:\n\n"
                f"Context:\n{context_string}\n\n"
                f"User Query: {user_query}\n\n"
                f"Answer:"
            )
            return self.generate_with_cohere(user_query, prompt)
        except Exception as e:
            logging.error(f"Error building final prompt: {e}")
            return "Sorry, something went wrong while preparing the answer."

    def chat_interface(self):
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

                local_docs = self.retrieve_local_documents(user_input)
                response = self.generate_final_response(user_input, local_docs)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nAssistant: Session ended. Goodbye!")
                break
            except EOFError:
                print("\n\nAssistant: Input stream closed. Exiting chat.")
                break
            except Exception as e:
                logging.error(f"Unexpected error in chat loop: {e}", exc_info=True)
                print("\nAssistant: Sorry, something went wrong. Try again.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        chatbot = HCLChatbot(data_folder="data", collection_name="hcl_docs_web_v2")
        chatbot.chat_interface()
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")








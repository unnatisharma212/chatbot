import os
import chromadb
from chromadb.utils import embedding_functions

def load_text_files(folder_path):
    docs = []
    ids = []
    metadatas = []

    try:
        files = os.listdir(folder_path)
        print(f"Found {len(files)} files in '{folder_path}'")
    except Exception as e:
        print(f"Error listing files in {folder_path}: {e}")
        return docs, ids, metadatas

    for i, filename in enumerate(files):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(content)
                    ids.append(f"doc_{i}")
                    metadatas.append({"source": filename})
                    print(f"Loaded file: {filename}")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return docs, ids, metadatas

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="my_docs",
        embedding_function=embedding_func
    )

    docs, ids, metas = load_text_files("data")

    if len(docs) == 0:
        print("No documents found to add!")
    else:
        collection.add(documents=docs, ids=ids, metadatas=metas)
        print(f"Added {len(docs)} documents to the ChromaDB collection 'my_docs'.")

except Exception as e:
    print(f"Error during ingestion: {e}")


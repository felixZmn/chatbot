import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import time
import warnings

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")

PERSIST_DIR = "./storage"


def load_index(directory):
    documents = SimpleDirectoryReader(
        directory, filename_as_id=True).load_data()

    try:
        # Load index from storage
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # Create index from documents and persist it
        print("Creating index from documents...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    # Refresh the reference documents and persist the index
    print("Refreshing reference documents...")
    index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


if __name__ == "__main__":
    # Start time
    start_time = time.time()

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Embeddings model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # Language model
    Settings.llm = Ollama(
        model="llama3.1", request_timeout=360.0, device=device)

    # Load index with documents from ./data
    index = load_index("data")

    # Perform RAG query
    print("Performing query...")
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "Was ist zur Abgabe der Bachelorarbeit notwendig?")
    print(response)

    # End time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

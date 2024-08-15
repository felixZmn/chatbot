import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import time
import os.path
import warnings

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")


# Start time
start_time = time.time()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# My local documents
documents = SimpleDirectoryReader("data").load_data()

# Embeddings model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Language model
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0, device=device)

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Perform RAG query
query_engine = index.as_query_engine()
response = query_engine.query(
    "Was ist zur Abgabe der Bachelorarbeit notwendig?")
print(response)

# End time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

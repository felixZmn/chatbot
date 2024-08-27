import time
import warnings

import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from ChatBot import ChatBot, Course


warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")


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

    # Set course
    course = Course.IT

    # setup Bot
    chat_bot = ChatBot()

    # Perform RAG query
    print("Performing query...")
    result = chat_bot.perform_query("Wer ist die Studiengangsleitung?", course)
    print(result)

    # Calculate and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")

    # Loop for chat
    while True:
        chat_bot.perform_query(input("\nFrage: "), course)
        time.sleep(1)

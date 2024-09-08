import time
import warnings

import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from ChatBot import ChatBot, Course
from logger import chatbot_logger


warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")


if __name__ == "__main__":
    # Loggers
    chatbot_logger = chatbot_logger(logLevel=10)

    # Start time
    start_time = time.time()

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chatbot_logger.info(f"Using device: {device}")

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
    query = "In welcher Straße befindet sich die DHBW?"
    result = chat_bot.perform_query(query, course)
    print(result)

    # Calculate and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    chatbot_logger.debug(f"Elapsed time: {elapsed_time:.2f} seconds")

    # # Loop for chat
    while True:
        query = input("\nFrage: ")
        result = chat_bot.perform_query(query, course)
        print(result)
        time.sleep(1)

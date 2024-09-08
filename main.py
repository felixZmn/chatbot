import time
import warnings


from ChatBot import ChatBot, Course
from logger import chatbot_logger, message_logger


warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")


if __name__ == "__main__":
    # Loggers
    chatbot_logger = chatbot_logger(logLevel=10)
    message_logger = message_logger(logLevel=10)

    # Start time
    start_time = time.time()

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

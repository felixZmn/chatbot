from src.logger import chatbot_logger, message_logger
from src.discord.DiscordBot import DiscordBot
import os


chatbot_logger = chatbot_logger(logLevel=10)
message_logger = message_logger(logLevel=10)

# environment variables
BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR')
INDEX_DIR = os.environ.get('INDEX_DIR')

if not BOT_TOKEN:
    chatbot_logger.error('DISCORD_BOT_TOKEN environment variable not set')
    exit(1)

if not DOCUMENTS_DIR:
    DOCUMENTS_DIR = './data/documents'

if not INDEX_DIR:
    INDEX_DIR = './data/index'

client = DiscordBot(documents_dir=DOCUMENTS_DIR, index_dir=INDEX_DIR)
client.run(BOT_TOKEN)

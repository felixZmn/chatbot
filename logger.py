import logging
import os

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _logger(logger: str, logLevel: int, file: str) -> logging.Logger:
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    handler = logging.FileHandler(
        filename=file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger)
    logger.setLevel(logLevel)
    logger.addHandler(handler)
    return logger


def chatbot_logger(logLevel: int = logging.WARNING, file: str = "temp/chatbot.log") -> logging.Logger:
    return _logger('ChatBot', logLevel, file)


def message_logger(logLevel: int = logging.DEBUG, file: str = "temp/messages.log") -> logging.Logger:
    """
    message logging enabled: logLevel = logging.INFO or logging.DEBUG
    message logging disabled: logLevel = logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    return _logger('Messages', logLevel, file)

import logging

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def chatbot_logger(logLevel: int = logging.WARNING, file: str = "temp/chatbot.log") -> logging.Logger:
    handler = logging.FileHandler(
        filename=file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger('ChatBot')
    logger.setLevel(logLevel)
    logger.addHandler(handler)
    return logger


def message_logger(logLevel: int = logging.DEBUG, file: str = "temp/messages.log") -> logging.Logger:
    """
    message logging enabled: logLevel = logging.INFO or logging.DEBUG
    message logging disabled: logLevel = logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    handler = logging.FileHandler(
        filename=file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger('Messages')
    logger.setLevel(logLevel)
    logger.addHandler(handler)
    return logger

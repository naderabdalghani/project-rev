import logging

MODELS_DIR = '.\models'
CACHE_DIR = '.\cache'
DATA_DIR = '.\data'
DEBUG = True
BOT_NAME = 'Ted'
TEXT_CHAT_MODE = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

import logging

import torch

MODELS_DIR = '.\models'
CACHE_DIR = '.\cache'
DATA_DIR = '.\data'
INPUT_FILENAME = "input.wav"
OUTPUT_FILENAME = "output.wav"
DEBUG = True
BOT_NAME = 'Ted'
TEXT_CHAT_MODE = False
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

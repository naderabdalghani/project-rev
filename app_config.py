import logging
import os

import torch

MODELS_DIR = '.\models'
CACHE_DIR = '.\cache'
DATA_DIR = '.\data'
INPUT_FILENAME = "input.wav"
OUTPUT_FILENAME = "output.wav"
DEBUG = True
BOT_NAME = 'Ted'
APP_MODE = "VOICE_CHAT_LITE_MODE"
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

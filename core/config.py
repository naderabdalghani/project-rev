import logging
import os.path

import torch

MODEL_NAME = 'facebook/blenderbot-400M-distill'
MODELS_DIR = '..\models'
CACHE_DIR = '..\cache'
DATA_DIR = '..\data'
DO_TRAIN = True
DO_EVAL = True
EVALUATE_DURING_TRAINING = False
PER_GPU_TRAIN_BATCH_SIZE = 4
PER_GPU_EVAL_BATCH_SIZE = 4
NUM_BATCHES_TILL_GRADIENT_ACCUMULATION = 1
EVAL_DATA_SPLIT_RATIO = 0.15
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
NUM_TRAIN_EPOCHS = 3
MAX_STEPS = -1  # If > 0, sets total number of training steps to perform. Overrides NUM_TRAIN_EPOCHS
WARMUP_STEPS = 0
LOGGING_STEPS = 50
SAVE_STEPS = 500
MAX_CHECKPOINTS = 2  # Maximum number of checkpoints. Older checkpoints get deleted if the number exceeds this value
OVERWRITE_CACHE = False
RESUME_TRAINING = False
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
BOT_TOKEN = '<bot>'
USER_TOKEN = '<s>'
with open(os.path.join(DATA_DIR, "BAD_WORDS.txt")) as f:
    BAD_WORDS = f.read().splitlines()
DATASET_FILENAME = 'HIMYM_DATASET.txt'
NO_DECAY_PARAMS_NAMES = ["bias", "ln"]
CHECKPOINT_PREFIX = "core-checkpoint"
SAVED_INSTANCE_PREFIX = "core-model"
LOSS_FN_IGNORE_INDEX = -100

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

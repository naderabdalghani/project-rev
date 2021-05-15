import logging
# Core module
OUTPUT_DIR = '../output'
MODEL_TYPE = 'gpt2'
MODEL_NAME_OR_PATH = 'microsoft/DialoGPT-small'
CONFIG_NAME = 'microsoft/DialoGPT-small'
TOKENIZER_NAME = 'microsoft/DialoGPT-small'
CACHE_DIR = '../cache'
BLOCK_SIZE = 512
DO_TRAIN = True
DO_EVAL = True
EVALUATE_DURING_TRAINING = False
PER_GPU_TRAIN_BATCH_SIZE = 4
PER_GPU_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
NUM_TRAIN_EPOCHS = 3
MAX_STEPS = -1
WARMUP_STEPS = 0
LOGGING_STEPS = 1000
SAVE_STEPS = 3500
SAVE_TOTAL_LIMIT = None
EVAL_ALL_CHECKPOINTS = False
NO_CUDA = True
OVERWRITE_OUTPUT_DIR = True
OVERWRITE_CACHE = True
SHOULD_CONTINUE = False
SEED = 42
LOCAL_RANK = -1
FP16 = False
FP16_OPT_LEVEL = 'O1'
BOT_TOKEN = '<|bot|>'
USER_TOKEN = '<|user|>'
SPECIAL_TOKENS_DICT = {'additional_special_tokens': [BOT_TOKEN, USER_TOKEN]}
DIALOGUE_SIZE = 10
DATASET_FILENAME = 'dataset.txt'
# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if LOCAL_RANK in [-1, 0] else logging.WARN,
)

# Flask
DEBUG = True

import torch
import logging

# Core module
OUTPUT_DIR = '.\output'  # Relative to the main file (app entry script)
CACHE_DIR = '.\cache'  # Relative to the main file (app entry script)
MODEL_TYPE = 'gpt2'
MODEL_NAME = 'microsoft/DialoGPT-small'
MODEL_CONFIG_NAME = 'microsoft/DialoGPT-small'
TOKENIZER_NAME = 'microsoft/DialoGPT-small'
DO_TRAIN = True
DO_EVAL = True
EVALUATE_DURING_TRAINING = False
PER_GPU_TRAIN_BATCH_SIZE = 4
PER_GPU_EVAL_BATCH_SIZE = 4
NUM_BATCHES_TILL_GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
NUM_TRAIN_EPOCHS = 3
MAX_STEPS = -1  # If > 0, sets total number of training steps to perform. Overrides NUM_TRAIN_EPOCHS
WARMUP_STEPS = 0
LOGGING_STEPS = 1000
SAVE_STEPS = 3500
MAX_CHECKPOINTS = 2  # Maximum number of checkpoints. Older checkpoints get deleted if the number exceeds this value
OVERWRITE_OUTPUT_DIR = True
OVERWRITE_CACHE = True
LOCAL_RANK = -1  # Distributed training local rank of process. -1 implies no distributed training
CUDA = torch.cuda.is_available()
if LOCAL_RANK != -1:
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device("cuda" if CUDA else "cpu", LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl')
else:
    DEVICE = torch.device("cuda" if CUDA else "cpu")
N_GPUS = torch.cuda.device_count() if CUDA else 0
TRAIN_BATCH_SIZE = PER_GPU_TRAIN_BATCH_SIZE * max(1, N_GPUS)
EVAL_BATCH_SIZE = PER_GPU_EVAL_BATCH_SIZE * max(1, N_GPUS)
FP16 = False  # Whether to use 16-bit (mixed) precision training through NVIDIA apex
FP16_OPT_LEVEL = 'O1'  # Apex fp16 training AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']
# See details at https://nvidia.github.io/apex/amp.html"
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

import torch
import logging

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
LOGGING_STEPS = 2
SAVE_STEPS = 2
MAX_CHECKPOINTS = 2  # Maximum number of checkpoints. Older checkpoints get deleted if the number exceeds this value
OVERWRITE_CACHE = False
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
BOT_TOKEN = '<bot>'
USER_TOKEN = '<s>'
with open("D:\Projects\project-rev\data\BAD_WORDS.txt", "r") as f:
    BAD_WORDS = f.read().splitlines()
DATASET_FILENAME = 'HIMYM_DATASET.txt'
NO_DECAY_PARAMS_NAMES = ["bias", "ln"]
CHECKPOINT_PREFIX = "core-checkpoint"
SAVED_INSTANCE_PREFIX = "core-model"
LOSS_FN_IGNORE_INDEX = -100

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if LOCAL_RANK in [-1, 0] else logging.WARN,
)

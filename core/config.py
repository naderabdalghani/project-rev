import os.path

from ray import tune

from app_config import DATA_DIR

MODEL_NAME = 'facebook/blenderbot-400M-distill'
HYPER_PARAMS_TUNING = True
TRIAL_NAME = ""
VALIDATE_WHILE_TRAINING = True
DO_TRAIN = True
DO_VALID = False
NUM_SAMPLES = 20
MAX_NUM_STEPS = 10000
MIN_NUM_STEPS = 1000
VALID_DATA_SPLIT_RATIO = 0.15
MAX_STEPS = -1  # If > 0, sets total number of training steps to perform. Overrides NUM_TRAIN_EPOCHS
LOGGING_STEPS = 50
SAVE_STEPS = 0
MAX_CHECKPOINTS = 2  # Maximum number of checkpoints. Older checkpoints get deleted if the number exceeds this value
OVERWRITE_CACHE = False
RESUME_TRAINING = False
HYPER_PARAMS = {
    "LEARNING_RATE": tune.qloguniform(1e-4, 1e-3, 1e-4),
    "WEIGHT_DECAY": tune.quniform(0, 0.25, 0.01),
    "ADAM_EPSILON": tune.qloguniform(1e-9, 1e-8, 1e-9),
    "NUM_TRAIN_EPOCHS": 5,
    "WARMUP_STEPS": tune.qrandint(0, 6000, 1000),
    "BATCH_SIZE": 2
}
DEFAULT_HYPER_PARAMS = {
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 0.0,
    "ADAM_EPSILON": 1e-8,
    "NUM_TRAIN_EPOCHS": 5,
    "WARMUP_STEPS": 0,
    "BATCH_SIZE": 4
}
MAX_GRAD_NORM = 1.0
NO_DECAY_PARAMS_NAMES = ["bias", "ln"]
BOT_TOKEN = '<bot>'
USER_TOKEN = '<s>'
DATASET_FILENAME = 'HIMYM_DATASET.txt'
CHECKPOINT_PREFIX = "core-checkpoint"
SAVED_INSTANCE_PREFIX = "core-model"
LOSS_FN_IGNORE_INDEX = -100
AVOID_BAD_WORDS = False
USE_BUILTIN_GENERATOR = True
USE_HISTORY = False

with open(os.path.join(DATA_DIR, "BAD_WORDS.txt")) as f:
    BAD_WORDS = f.read().splitlines()

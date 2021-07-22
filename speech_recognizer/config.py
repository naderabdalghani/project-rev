from .text_transformer import TextTransformer
import torch
import os

from app_config import DATA_DIR

FEATURE_USED = {'name': 'mel-spectrogram', 'n_features': 128}
SAMPLING_RATE = 16000
TEXT_TRANSFORMER = TextTransformer()
HYPER_PARAMS = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": FEATURE_USED['n_features'],
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "batch_size": 16,
    "epochs": 20
}
NUM_WORKERS = 4
TRAIN_DATASET_URL = "train-clean-360"
VALID_DATASET_URL = "test-clean"
SAVED_INSTANCE_NAME = "speech-recognizer-{}.pt".format(FEATURE_USED['name'])
LOGGING_STEPS = 2000
COMMON_VOICE_TSV_FILE_PATH = os.path.join(DATA_DIR, 'cv-corpus-6.1-2020-12-11/en/validated.tsv')  # Contains file path of validate.tsv
CREATED_JSON_PATH = os.path.join(DATA_DIR, 'cv-corpus-6.1-2020-12-11/en')  # Contains directory for json files
NUM_OF_PROCESSES = 30  # Number of processes to be run for mp3 conversion in case of multiprocessing
NUM_OF_THREADS = 5  # Number of threads to be run for mp3 conversion in case of multithreading
AUDIO_FILE_MAX_DURATION = 10

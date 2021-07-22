from text_transformer import TextTransformer
import torch

MODELS_DIR = '..\models'
DATA_DIR = '..\data'
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
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
NUM_WORKERS = 4
TRAIN_DATASET_URL = "train-clean-360"
VALID_DATASET_URL = "test-clean"
SAVED_INSTANCE_NAME = "speech-recognizer-{}.pt".format(FEATURE_USED['name'])
LOGGING_STEPS = 2000

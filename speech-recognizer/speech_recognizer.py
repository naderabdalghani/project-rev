import os

from comet_ml import Experiment
import torchaudio
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tqdm import trange

from exceptions import SpeechRecognizerNotTrained
from keys import COMET_API_KEY
from config import HYPER_PARAMS, DATA_DIR, CUDA, NUM_WORKERS, TRAIN_DATASET_URL, VALID_DATASET_URL, DEVICE, MODELS_DIR, \
    SAVED_INSTANCE_NAME
from preprocessing import preprocess, valid_audio_transforms
from model import SpeechRecognitionModel
from text_transformer import TextTransformer
from train import train, validate
from utils import greedy_decode
import torch.nn.functional as F

loaded_model = None


def load_saved_instance():
    global loaded_model
    saved_instance_path = os.path.join(MODELS_DIR, SAVED_INSTANCE_NAME)
    if os.path.isfile(saved_instance_path):
        checkpoint = torch.load(saved_instance_path, map_location=DEVICE)
        hyper_params = checkpoint['hparams']
        loaded_model = SpeechRecognitionModel(
            hyper_params['n_cnn_layers'], hyper_params['n_rnn_layers'], hyper_params['rnn_dim'],
            hyper_params['n_class'], hyper_params['n_feats'], hyper_params['stride'], hyper_params['dropout']
        ).to(DEVICE)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()
    else:
        raise SpeechRecognizerNotTrained()


@torch.no_grad()
def wav_to_text():
    global loaded_model
    if loaded_model is None:
        load_saved_instance()
    waveform, _ = torchaudio.load(os.path.join(DATA_DIR, 'test1.wav'))
    spectrogram = valid_audio_transforms(waveform).unsqueeze(0)
    decoded_predictions, _ = greedy_decode(F.log_softmax(loaded_model(spectrogram), dim=2))
    print(decoded_predictions)


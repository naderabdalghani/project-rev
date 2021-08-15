import torch
import torchaudio
import torch.nn as nn

from app_config import DEVICE
from .config import FEATURE_USED, SAMPLING_RATE, TEXT_TRANSFORMER

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, n_mels=FEATURE_USED['n_features']).to(DEVICE)
    if FEATURE_USED['name'] == 'mel-spectrogram' else torchaudio.transforms.MFCC(sample_rate=SAMPLING_RATE,
                                                                                 n_mfcc=FEATURE_USED['n_features'])
    .to(DEVICE),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30).to(DEVICE),
    torchaudio.transforms.TimeMasking(time_mask_param=100).to(DEVICE)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,
                                                              n_mels=FEATURE_USED['n_features']).to(DEVICE)\
    if FEATURE_USED['name'] == 'mel-spectrogram' else torchaudio.transforms.MFCC(sample_rate=SAMPLING_RATE,
                                                                                 n_mfcc=FEATURE_USED['n_features'])\
    .to(DEVICE)


def preprocess(data, data_type="train"):
    """
    This function do the preprocessing phase for raw audio data.
    :param data: Audio files from the dataset
    :param data_type: Train or validation
    :return: spectrograms, labels, input_lengths, label_lengths
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(TEXT_TRANSFORMER.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

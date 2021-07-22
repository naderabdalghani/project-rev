import torchaudio
import torch.nn as nn

from config import FEATURE_USED, SAMPLING_RATE

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, n_mels=FEATURE_USED['n_features'])
    if FEATURE_USED['name'] == 'mel-spectrogram' else torchaudio.transforms.MFCC(sample_rate=SAMPLING_RATE,
                                                                                 n_mfcc=FEATURE_USED['n_features']),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,
                                                              n_mels=FEATURE_USED['n_features'])\
    if FEATURE_USED['name'] == 'mel-spectrogram' else torchaudio.transforms.MFCC(sample_rate=SAMPLING_RATE,
                                                                                 n_mfcc=FEATURE_USED['n_features'])



"""
Utility script to get mfcc features and store it in the disk.
This script is run after running mp3_dataset_converter script
"""
import multiprocessing
import json
import pickle
import torch.nn as nn
import torchaudio
import os
from tqdm import tqdm
from ..config import CREATED_JSON_PATH, NUM_OF_PROCESSES, AUDIO_FILE_MAX_DURATION
from app_config import CACHE_DIR


CACHE_FILE = os.path.join(CACHE_DIR, 'speech_recognizer_features.pickle')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MFCC(sample_rate=48000, n_mfcc=16),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MFCC(sample_rate=48000, n_mfcc=16)


def save_features():
    """ This function is called once after the json files is created
    """
    audio_samples = []
    dataset_types = ['train', 'valid', 'test']

    for dataset_type in dataset_types:
        if dataset_type == 'train':
            with open(CREATED_JSON_PATH + dataset_type + '_corpus.json') as json_line_file:
                for json_line in json_line_file:
                    audio_sample = json.loads(json_line)
                    audio_samples.append(audio_sample)
            extract_dataset_features(audio_samples, dataset_type)
            audio_samples = []
        else:
            with open(CREATED_JSON_PATH + dataset_type + '_corpus.json') as json_line_file:
                for json_line in json_line_file:
                    audio_sample = json.loads(json_line)
                    audio_samples.append(audio_sample)
    extract_dataset_features(audio_samples, 'test')


def extract_dataset_features(audio_samples, dataset_type):
    # Split data to be run over multi processes
    arguments = []
    spectrograms = []
    argument_len = len(audio_samples) // NUM_OF_PROCESSES
    for p in range(NUM_OF_PROCESSES):
        start_idx = p * argument_len
        end_idx = (p + 1) * argument_len
        if p == NUM_OF_PROCESSES - 1:
            end_idx = len(audio_samples)
        arguments.append((audio_samples[start_idx:end_idx], dataset_type))

    # Run data with process equals to number of processes
    with multiprocessing.Pool(NUM_OF_PROCESSES) as p:
        for processed_batch in p.starmap(extract_audio_sample_features, arguments):
            spectrograms += processed_batch

    print("----------------- " + dataset_type.capitalize() + "ing dataset conversion done! ------------------")
    with open(os.path.join(CREATED_JSON_PATH, dataset_type + '_' + CACHE_FILE), 'wb') as handle:
        pickle.dump(spectrograms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumping done!")


def extract_audio_sample_features(audio_samples, dataset_type):
    spectrograms = []
    process_name = multiprocessing.current_process().name
    if dataset_type == 'train':
        for audio_sample in tqdm(audio_samples, desc=process_name, leave=True, position=0):
            if audio_sample['duration'] < AUDIO_FILE_MAX_DURATION:
                waveform, _ = torchaudio.load(audio_sample["key"])
                spectrograms.append((train_audio_transforms(waveform).squeeze(0).transpose(0, 1).tolist(),
                                     audio_sample['text']))
    else:
        for audio_sample in tqdm(audio_samples, desc=process_name, leave=True, position=0):
            if audio_sample['duration'] < AUDIO_FILE_MAX_DURATION:
                waveform, _ = torchaudio.load(audio_sample["key"])
                spectrograms.append((valid_audio_transforms(waveform).squeeze(0).transpose(0, 1).tolist(),
                                     audio_sample['text']))
    return spectrograms


if __name__ == "__main__":
    save_features()
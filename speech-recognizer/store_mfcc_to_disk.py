""" Utility script to get mfcc features and store it in the disk
"""
import multiprocessing
import json
from tqdm.notebook import tqdm
import pickle
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
from tqdm import tqdm
import random
import numpy as np

JSON_PATH = 'data/cv-corpus-6.1-2020-12-11/en/'  # Contains directory for json files
NUM_OF_PROCESSES = 30
MFCC_DIM = 13
RNG_SEED = 123
MAX_DURATION = 10
PICKLE_NAME = 'features'

mean = 0
std = 1


def store_mfcc_to_disk():
    """ This function is called once after the json files is created
    """
    features = []
    types = ['train', 'valid', 'test']
    specs = []
    cnt = {'train': 0, 'valid': 0, 'test': 0}
    # Make mfcc directory, if necessary
    if not os.path.exists(JSON_PATH + '/mfcc'):
        os.makedirs(JSON_PATH + '/mfcc')

    for type in types:
        with open(JSON_PATH + type + '_corpus.json') as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    specs.append(spec)
                    cnt[type] += 1
                except Exception as e:
                    print('Error reading line #{}: {}'
                          .format(line_num, json_line))
    mean, std = get_mean_std(specs)
    # Split data to be run over multi processes
    arguments = []
    argument_len = len(specs) // NUM_OF_PROCESSES
    for p in range(NUM_OF_PROCESSES):
        start_idx = p * argument_len
        end_idx = (p + 1) * argument_len
        if p == NUM_OF_PROCESSES - 1:
            end_idx = len(specs)
        arguments.append(specs[start_idx:end_idx])

    # Run data with process equals to number of processes
    with multiprocessing.Pool(NUM_OF_PROCESSES) as p:
        for result in p.imap_unordered(create_mfcc, arguments):
            features += result

    print("-----------------All converting Done!------------------")
    with open(JSON_PATH + PICKLE_NAME, 'wb') as mf:
        pickle.dump(features, mf, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")


def featurize(audio_clip):
    """ For a given audio clip, calculate the corresponding feature
        :param audio_clip: (str) Path to the audio clip
        :returns: MFCC
    """
    (rate, sig) = wav.read(audio_clip)
    return mfcc(sig, rate, nfft=1200, numcep=MFCC_DIM)


def get_mean_std(data, k_samples=2000):
    """ Estimate the mean and std of the features from the training set
        :param data: data that contains path of the wav audio
        :param k_samples: (int) Use this number of samples for estimation
        :returns feats_mean: (float) Mean of the k_sample
        :returns feats_std: (float) Std of the k_sample
    """
    k_samples = min(k_samples, len(data))
    rng = random.Random(RNG_SEED)
    audio_paths = [row['key'] for row in data]
    samples = rng.sample(audio_paths, k_samples)
    feats = [featurize(s) for s in samples]
    feats = np.vstack(feats)
    feats_mean = np.mean(feats, axis=0)
    feats_std = np.std(feats, axis=0)
    return feats_mean, feats_std


def normalize(feature, mean, std, eps=1e-14):
    """ Center a feature using the mean and std
        :param feature: (numpy.ndarray) Feature to normalize
        :param mean: (float) Mean of features
        :param std: (float) Std of features
        :param eps: (float) Small float number to avoid divide by zero
        :returns: The normalized features
    """
    return (feature - mean) / (std + eps)


def create_mfcc(specs):
    features = []
    process_name = multiprocessing.current_process().name
    for row in tqdm(specs, desc=process_name, leave=True, position=0):
        if row['duration'] <= MAX_DURATION:
            features.append((normalize(featurize(row['key']), mean, std)), row['text'])
    return features


if __name__ == "__main__":
    store_mfcc_to_disk()
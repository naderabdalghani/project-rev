""" Utility script to get mfcc features and store it in the disk
"""
import multiprocessing
import json
import pickle

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
from tqdm import tqdm

JSON_PATH = 'data/cv-corpus-6.1-2020-12-11/en'  # Contains directory for json files
NUM_OF_PROCESSES = 30
MFCC_DIM = 13

def store_mfcc_features_to_disk():
    """ This function is called once after the json files is created
    """
    types = ['train', 'valid', 'test']
    specs = []
    cnt = {'train': 0, 'valid': 0, 'test': 0}
    # Make mfcc directory, if necessary
    if not os.path.exists(JSON_PATH + '/mfcc'):
        os.makedirs(JSON_PATH + '/mfcc')

    while type in types:
        with open(JSON_PATH + type + '_corpus.json') as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    specs.append(spec)
                    cnt[type] += 1
                except Exception as e:
                    print('Error reading line #{}: {}'
                          .format(line_num, json_line))

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
            specs += result


def featurize(audio_clip):
    """ For a given audio clip, calculate the corresponding feature
        :param audio_clip: (str) Path to the audio clip
        :returns: MFCC
    """
    (rate, sig) = wav.read(audio_clip)
    return mfcc(sig, rate, nfft=1200, numcep=MFCC_DIM)


def create_mfcc(specs):
    data = []
    process_name = multiprocessing.current_process().name
    for row in tqdm(specs, desc=process_name, leave=True, position=0):
        mfcc_path = row['path']
        mfcc_path = mfcc_path.replace('clips', 'mfcc')
        with open(mfcc_path, 'wb') as handle:
            pickle.dump(featurize(row['path']), handle, protocol=pickle.HIGHEST_PROTOCOL)
        row['path'] = mfcc_path
        data.append(row)
    return data

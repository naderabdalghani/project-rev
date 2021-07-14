import json
import numpy as np
import random
import librosa
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence

RNG_SEED = 123


class AudioGenerator:
    def __init__(self, step=10, window=20, max_freq=8000, mfcc_dim=13,
                 minimum_batch_size=20, desc_file=None, spectrogram=True, max_duration=10.0,
                 sort_by_duration=False):
        """ Generates training, validation and testing data
            :param step: (int) Step size in milliseconds between windows (for spectrogram ONLY)
            :param window: (int) FFT window size in milliseconds (for spectrogram ONLY)
            :param max_freq: (int) Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned (for spectrogram ONLY)
            :param desc_file: (str, optional) Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """

        self.feat_dim = calc_feat_dim(window, max_freq)
        self.mfcc_dim = mfcc_dim
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.max_duration = max_duration
        self.minimum_batch_size = minimum_batch_size
        self.spectrogram = spectrogram
        self.sort_by_duration = sort_by_duration

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
            :param partition: (string) Chooses from ('train', 'valid', 'test')
            :raises: Exception if partition has a value different from ('train', 'valid', 'test')
            :returns inputs: Contains the wav audio, label, input length and label length
        """
        if partition == 'train':
            audio_paths = self.train_audio_paths
            cur_index = self.cur_train_index
            texts = self.train_texts
        elif partition == 'valid':
            audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            texts = self.valid_texts
        elif partition == 'test':
            audio_paths = self.test_audio_paths
            cur_index = self.test_valid_index
            texts = self.test_texts
        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation or test")

        features = [self.normalize(self.featurize(a)) for a in
                    audio_paths[cur_index:cur_index + self.minimum_batch_size]]

        # Calculate necessary sizes
        max_length = max([features[i].shape[0]
                          for i in range(0, self.minimum_batch_size)])
        max_string_length = max([len(texts[cur_index + i])
                                 for i in range(0, self.minimum_batch_size)])

        # Initialize the arrays
        input_data = np.zeros([self.minimum_batch_size, max_length,
                               self.feat_dim * self.spectrogram + self.mfcc_dim * (not self.spectrogram)])
        labels = np.ones([self.minimum_batch_size, max_string_length]) * 28  # Set all labels as blank
        input_length = np.zeros([self.minimum_batch_size, 1])
        label_length = np.zeros([self.minimum_batch_size, 1])

        for i in range(0, self.minimum_batch_size):
            # Calculate input_data & input_length
            feat = features[i]
            input_length[i] = feat.shape[0]
            input_data[i, :feat.shape[0], :] = feat

            # Calculate labels & label_length
            label = np.array(text_to_int_sequence(texts[cur_index + i]))
            labels[i, :len(label)] = label
            label_length[i] = len(label)

        # Return the arrays
        outputs = {'ctc': np.zeros([self.minimum_batch_size])}
        inputs = {'the_input': input_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length
                  }
        return inputs, outputs

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the corresponding feature
            :param audio_clip: (str) Path to the audio clip
            :returns: Spectrogram or MFCC
        """
        if self.spectrogram:
            return spectrogram_from_file(
                audio_clip, step=self.step, window=self.window,
                max_freq=self.max_freq)
        else:
            (rate, sig) = wav.read(audio_clip)
            return mfcc(sig, rate, numcep=self.mfcc_dim)

    def normalize(self, feature, eps=1e-14):
        """ Center a feature using the mean and std
            :param feature: (numpy.ndarray) Feature to normalize
            :returns: The normalized features
        """
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        :param partition: (str) train or valid
        :raises: Exception if the partition was not train/valid
        :returns: None
        """
        if partition == 'train':
            self.train_audio_paths, self.train_durations, self.train_texts = shuffle_data(
                self.train_audio_paths, self.train_durations, self.train_texts)
        elif partition == 'valid':
            self.valid_audio_paths, self.valid_durations, self.valid_texts = shuffle_data(
                self.valid_audio_paths, self.valid_durations, self.valid_texts)
        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation")

    def sort_data_by_duration(self, partition):
        """ Sort the training or validation sets by (increasing) duration
            :param partition: (str) train or valid
            :raises: Exception if the partition was not train/valid
            :returns: None
        """
        if partition == 'train':
            self.train_audio_paths, self.train_durations, self.train_texts = sort_data(
                self.train_audio_paths, self.train_durations, self.train_texts)
        elif partition == 'valid':
            self.valid_audio_paths, self.valid_durations, self.valid_texts = sort_data(
                self.valid_audio_paths, self.valid_durations, self.valid_texts)
        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation")

    def next_train(self):
        """ Obtain a batch of training data
            :returns: Batch of training data
        """
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += self.minimum_batch_size
            if self.cur_train_index >= len(self.train_texts) - self.minimum_batch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret

    def next_valid(self):
        """ Obtain a batch of validation data
            :returns: Batch of validation data
        """
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += self.minimum_batch_size
            if self.cur_valid_index >= len(self.valid_texts) - self.minimum_batch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def next_test(self):
        """ Obtain a batch of test data
            :returns: Batch of testing data
        """
        while True:
            ret = self.get_batch('test')
            self.cur_test_index += self.minimum_batch_size
            if self.cur_test_index >= len(self.test_texts) - self.minimum_batch_size:
                self.cur_test_index = 0
            yield ret

    def load_train_data(self, desc_file='train.json'):
        self.load_metadata_from_desc_file(desc_file, 'train')
        self.fit_train()
        if self.sort_by_duration:
            self.sort_data_by_duration('train')

    def load_validation_data(self, desc_file='valid.json'):
        self.load_metadata_from_desc_file(desc_file, 'validation')
        if self.sort_by_duration:
            self.sort_data_by_duration('valid')

    def load_test_data(self, desc_file='test.json'):
        self.load_metadata_from_desc_file(desc_file, 'test')

    def load_metadata_from_desc_file(self, desc_file, partition):
        """ Read metadata from a JSON file, sets paths, duration, texts based on partition
            :param desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            :param partition (str): One of 'train', 'validation' or 'test'
            :raises: Exception if it can not read a line or a file
        """
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > self.max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    print('Error reading line #{}: {}'
                          .format(line_num, json_line))
        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.valid_audio_paths = audio_paths
            self.valid_durations = durations
            self.valid_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")


def shuffle_data(audio_paths, durations, texts):
    """ Shuffle the data (called after making a complete pass through
        training or validation data during the training process)
        :param audio_paths: (list) Paths to audio clips
        :param durations: (list) Durations of utterances for each audio clip
        :param texts: (list) Sentences uttered in each audio clip
        :returns: Shuffled data with paths, duration and texts
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p]
    durations = [durations[i] for i in p]
    texts = [texts[i] for i in p]
    return audio_paths, durations, texts


def sort_data(audio_paths, durations, texts):
    """ Sort the data by duration
        :param audio_paths: (list) Paths to audio clips
        :param durations: (list) Durations of utterances for each audio clip
        :param texts: (list) Sentences uttered in each audio clip
        :returns: Sorted data with paths, duration and texts
    """
    p = np.argsort(durations).tolist()
    audio_paths = [audio_paths[i] for i in p]
    durations = [durations[i] for i in p]
    texts = [texts[i] for i in p]
    return audio_paths, durations, texts


def vis_train_features(index=0):
    """ Visualizing the data point in the training set at the supplied index
        :param index: (int) Index of data to be visualized
        :returns: vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path
    """
    # Obtain spectrogram
    audio_gen = AudioGenerator(spectrogram=True)
    audio_gen.load_train_data()
    vis_audio_path = audio_gen.train_audio_paths[index]
    vis_spectrogram_feature = audio_gen.normalize(audio_gen.featurize(vis_audio_path))
    # Obtain mfcc
    audio_gen = AudioGenerator(spectrogram=False)
    audio_gen.load_train_data()
    vis_mfcc_feature = audio_gen.normalize(audio_gen.featurize(vis_audio_path))
    # Obtain text label
    vis_text = audio_gen.train_texts[index]
    # Obtain raw audio
    vis_raw_audio, _ = librosa.load(vis_audio_path)
    # Print total number of training examples
    print('There are %d total training examples.' % len(audio_gen.train_audio_paths))
    # Return labels for plotting
    return vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path


def plot_raw_audio(vis_raw_audio):
    """ Visualize audio in a wave form
    :param vis_raw_audio: Audio data to be plotted
    """
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    steps = len(vis_raw_audio)
    ax.plot(np.linspace(1, steps, steps), vis_raw_audio)
    plt.title('Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_mfcc_feature(vis_mfcc_feature):
    """ Plot mfcc feature of the audio
    :param vis_mfcc_feature: MFCC of the audio
    """
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(0, 13, 2), minor=False);
    plt.show()


def plot_spectrogram_feature(vis_spectrogram_feature):
    """ Plot the normalized spectrogram
    :param vis_spectrogram_feature: spectrogram of the audio
    """
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized Spectrogram')
    plt.ylabel('Time')
    plt.xlabel('Frequency')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
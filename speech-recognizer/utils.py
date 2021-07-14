"""
Defines various functions for processing the data.
"""
import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided
from char_map import char_map, index_map


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time.
    Params:
        :param input_length: (int) Length of the input sequence.
        :param filter_size: (int) Width of the convolution kernel.
        :param border_mode: (str) Only support `same` or `valid`.
        :param stride: (int) Stride size used in 1D convolution.
        :param dilation: (int)
        :returns output_length: After the 1D convolution
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """ Compute the spectrogram for a real signal.
    Args:
        :param samples: (1D array) input audio signal
        :param fft_length: (int) number of elements in fft window
        :param sample_rate: (scalar) sample rate
        :param hop_length: (int) hop length (relative offset between neighboring
            fft windows).

    Returns:
        :returns x: (2D array) spectrogram [frequency x time]
        :returns freq: (1D array) frequency of each row in x
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"
    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)
    scale = window_norm * sample_rate
    trunc = (len(samples) - fft_length) % hop_length  # Truncate the rest
    x = samples[:len(samples) - trunc]
    # Reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)
    # Window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])
    # Broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2
    # Scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale
    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])
    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
        :param filename: (str) Path to the audio file
        :param step: (int) Step size in milliseconds between windows
        :param window: (int) FFT window size in milliseconds
        :param max_freq: (int) Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        :param eps: (float) Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        # Nyquist rate
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence
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

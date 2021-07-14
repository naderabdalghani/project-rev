"""
Defines various functions for processing the data.
"""
import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided
from char_map import char_map, index_map


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


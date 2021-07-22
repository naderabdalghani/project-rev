import numpy as np
import torch

from config import TEXT_TRANSFORMER


def get_levenshtein_distance(reference, hypothesis):
    """
    Levenshtein distance or edit distance is a metric for measuring the difference
    between two sequences. Levenshtein distance can be measured as the minimum number
    of single-character edits (insertions, deletions, or substitution)
    :param reference: (string) The reference sentence.
    :param hypothesis: (string) The hypothesis sentence.
    :return: (int) Levenshtein distance
    """
    len_max_string = len(reference)
    len_min_string = len(hypothesis)

    # Special cases
    if reference == hypothesis:
        return 0
    if len_max_string == 0:
        return len_min_string
    if len_min_string == 0:
        return len_max_string

    # Store the longest string to reference and the other one to hypothesis
    if len_max_string < len_min_string:
        reference, hypothesis = hypothesis, reference
        len_max_string, len_min_string = len_min_string, len_max_string

    # Create matrix of zeros with dimensions = 2 * len_min_string
    distance = np.zeros((2, len_min_string + 1), dtype=np.int32)

    # Initialize distance matrix
    for j in range(0, len_min_string + 1):
        distance[0][j] = j

    # Calculate levenshtein distance
    for i in range(1, len_max_string + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, len_min_string + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1  # Substitute
                i_num = distance[cur_row_idx][j - 1] + 1  # Insertion
                d_num = distance[prev_row_idx][j] + 1  # Deletion
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[len_max_string % 2][len_min_string]


def calculate_word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """
    Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: (string) The reference sentence.
    :param hypothesis: (string) The hypothesis sentence.
    :param ignore_case: (bool) Whether case-sensitive or not.
    :param delimiter: (char) Delimiter of input sentences.
    :return: (list (float, int)) Levenshtein distance and word number of reference sentence.
    """
    # Change reference and hypothesis to lower case in case of ignore_case
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    # Split on delimiter
    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = get_levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def calculate_character_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """
    Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: (string) The reference sentence.
    :param hypothesis: (string) The hypothesis sentence.
    :param ignore_case: (bool) Whether case-sensitive or not.
    :param remove_space: (bool) Whether remove internal space characters
    :return: (list (float, int)) Levenshtein distance and length of reference sentence.
    """
    # Change reference and hypothesis to lower case in case of ignore_case
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = get_levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def calculate_word_error_rate(reference, hypothesis, ignore_case=False, delimiter=' '):
    """
    Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    Equation:
        WER = (Sw + Dw + Iw) / Nw  =  levenshtein_distance / Nw
    where
        Sw is the number of words substituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
        Levenshtein distance = Sw + Dw + Iw
    We can use levenshtein distance to calculate WER.
    :param reference: (string) The reference sentence.
    :param hypothesis: (string) The hypothesis sentence.
    :param ignore_case: (bool) Whether case-sensitive or not.
    :param delimiter: (char) Delimiter of input sentences.
    :return: (float) Word error rate.
    :raises ValueError: If word reference length is zero.
    :return: (float) Word error rate
    """
    edit_distance, ref_len = calculate_word_errors(reference, hypothesis, ignore_case,
                                                   delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def calculate_character_error_rate(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate character error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    Equation:
        CER = (Sc + Dc + Ic) / Nc  =  levenshtein_distance / Nw
    where
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
        Levenshtein distance = Sc + Dc + Ic
    We can use levenshtein distance to calculate CER. Noting that spaces at the beginning
    and at the end of the sentences will be truncated, also many consecutive spaces will be
    replaced with only one space .
    :param reference: (String) The reference sentence.
    :param hypothesis: (String) The hypothesis sentence.
    :param ignore_case: (bool) Whether case-sensitive or not.
    :param remove_space: (bool) Whether remove internal space characters
    :raises ValueError: If the reference length is zero.
    :return: (float) Character error rate.
    """
    edit_distance, ref_len = calculate_character_errors(reference, hypothesis, ignore_case,
                                                        remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def greedy_decode(output, labels=None, label_lengths=None, blank_label=TEXT_TRANSFORMER.BLANK_LABEL,
                  collapse_repeated=True):
    """
    This function returns the char sequence using greedy approach given the output of the model
    :param output: (n * number_of_labels matrix) The output of the model (probability matrix)
    :param labels: (number_of_labels * m matrix) Labels, in our case m = 1
    :param label_lengths: (list) Length of each label
    :param blank_label: (int) The corresponding value of the blank label
    :param collapse_repeated: (bool) Ignore adding label if it was found at the index before thee current index
    :return: (list (list, list)) decoded values and target values
    """
    # Get the max probabilities
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        if labels is not None:
            targets.append(TEXT_TRANSFORMER.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(TEXT_TRANSFORMER.int_to_text(decode))
    return decodes, targets

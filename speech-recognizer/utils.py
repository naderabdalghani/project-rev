import numpy as np


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


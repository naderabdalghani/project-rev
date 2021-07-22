def get_levenshtein_distance(ref, hyp):
    """
    Levenshtein distance or edit distance is a metric for measuring the difference
    between two sequences. Levenshtein distance can be measured as the minimum number
    of single-character edits (insertions, deletions, or substitution)
    :param ref: (String) The first string to be compared with.
    :param hyp: (String) The second string to be compared with.
    :return: Levenshtein distance
    """
    len_max_string = len(ref)
    len_min_string = len(hyp)

    # Special cases
    if ref == hyp:
        return 0
    if len_max_string == 0:
        return len_min_string
    if len_min_string == 0:
        return len_max_string

    # Store the longest string to ref and the other one to hyp
    if len_max_string < len_min_string:
        ref, hyp = hyp, ref
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
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1  # Substitute
                i_num = distance[cur_row_idx][j - 1] + 1  # Insertion
                d_num = distance[prev_row_idx][j] + 1  # Deletion
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[len_max_string % 2][len_min_string]
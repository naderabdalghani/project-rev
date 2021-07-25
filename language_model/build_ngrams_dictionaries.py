import logging
import os
import pickle

import nltk

from app_config import DATA_DIR, MODELS_DIR
from .config import DATASET_FILENAME, UNIGRAMS_DICT_NAME, BIGRAMS_DICT_NAME, TRIGRAMS_DICT_NAME

logger = logging.getLogger(__name__)


def split_to_sentences(data):
    """
    Split data by linebreak "\n"

    Args:
        data: str

    Returns:
        A list of sentences
    """
    sentences = data.split("\n")

    # Additional cleaning
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences


def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """

    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []

    # Go through each sentence
    for sentence in sentences:
        # Convert to lowercase letters
        sentence = sentence.lower()

        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)

        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def get_tokenized_data(data):
    """
    Make a list of tokenized sentences

    Args:
        data: String

    Returns:
        List of lists of tokens
    """

    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)
    logger.info(len(sentences))
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)

    return tokenized_sentences


def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """

    word_counts = {}

    # Loop through each sentence
    for sentence in tokenized_sentences:

        # Go through each token in the sentence
        for token in sentence:

            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts.keys():
                word_counts[token] = 1

            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1

    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more

    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.

    Returns:
        List of words that appear N times or more
    """
    # Initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []

    # Get the word counts of the tokenized sentences
    # Use the function that you defined earlier to count the words
    word_counts = count_words(tokenized_sentences)

    # for each word and its count
    for word, cnt in word_counts.items():

        # check that the word's count
        # is at least as great as the minimum count
        if cnt >= count_threshold:
            # append the word to the list
            closed_vocab.append(word)

    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.

    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words

    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """

    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)

    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []

    # Go through each sentence
    for sentence in tokenized_sentences:

        # Initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []

        # for each token in the sentence
        for token in sentence:

            # Check if the token is in the closed vocabulary
            if token in vocabulary:
                # If so, append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append("<unk>")

        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


def preprocess_data(train_data, count_threshold):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.
    Args:
        train_data: List of lists of strings.
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """

    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)

    return train_data_replaced, vocabulary


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: number of words in a sequence
        start_token: start-of-sentence token
        end_token: end-of-sentence token

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    # Go through each sentence in the data
    for sentence in data:

        # prepend start token n-1 times, and  append <e> one time
        sentence = [start_token] * (n-1) + sentence + [end_token]
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)

        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.

        m = len(sentence) if n == 1 else len(sentence) - n + 1
        for i in range(m):

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i + n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams.keys():

                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1

    return n_grams


def build_dictionary():
    dataset_path = os.path.join(DATA_DIR, DATASET_FILENAME)
    if os.path.isfile(dataset_path):
        with open(os.path.join(DATA_DIR, DATASET_FILENAME), encoding='utf-8') as f:
            data = f.read()
        logger.info("Data type:", type(data))
        logger.info("Number of letters:", len(data))
        train_data = get_tokenized_data(data)
        minimum_freq = 2
        train_data_processed, vocabulary = preprocess_data(train_data, minimum_freq)
        unigrams = count_n_grams(train_data_processed, 1)
        bigrams = count_n_grams(train_data_processed, 2)
        trigrams = count_n_grams(train_data_processed, 3)
        pickle.dump(unigrams, open(os.path.join(MODELS_DIR, UNIGRAMS_DICT_NAME), 'wb'))
        pickle.dump(bigrams, open(os.path.join(MODELS_DIR, BIGRAMS_DICT_NAME), 'wb'))
        pickle.dump(trigrams, open(os.path.join(MODELS_DIR, TRIGRAMS_DICT_NAME), 'wb'))
        logger.info("Dictionaries Built Successfully")
    else:
        raise Exception("Dataset was not found please make sure that dataset is downloaded")


if __name__ == '__main__':
    build_dictionary()

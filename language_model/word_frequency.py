import pickle
from collections import Counter
import string
from utils import parse_into_words


class WordFrequency:
    def __init__(self):
        self._dictionary = Counter()
        self._total_words = 0
        self._unique_words = 0
        self._letters = set()
        self._longest_word_length = 0

        self._tokenizer = parse_into_words

    def __contains__(self, key):
        """ Check if dictionary contains key """
        key = key.lower()
        return key in self._dictionary

    def __getitem__(self, key):
        """ Get frequency """
        key = key.lower()
        return self._dictionary[key]

    def __iter__(self):
        """ iter support """
        for word in self._dictionary:
            yield word

    def pop(self, key, default=None):
        """ Delete the key and return the associated value or default if not
            found
            Args:
                key (str): The key to remove
                default (obj): The value to return if key is not present """
        key = key.lower()
        return self._dictionary.pop(key, default)

    @property
    def dictionary(self):
        """ Counter: A counting dictionary of all words in the corpus and the \
            number of times each has been seen
            Note:
                Not settable """
        return self._dictionary

    @property
    def total_words(self):
        """ int: The sum of all word occurrences in the word frequency \
                 dictionary
            Note:
                Not settable """
        return self._total_words

    @property
    def unique_words(self):
        """ int: The total number of unique words in the word frequency list
            Note:
                Not settable """
        return self._unique_words

    @property
    def letters(self):
        """ str: The listing of all letters found within the corpus
            Note:
                Not settable """
        return self._letters

    @property
    def longest_word_length(self):
        """ int: The longest word length in the dictionary
            Note:
                Not settable """
        return self._longest_word_length

    def keys(self):
        """ Iterator over the key of the dictionary
            Yields:
                str: The next key in the dictionary
            Note:
                This is the same as `spellchecker.words()` """
        for key in self._dictionary.keys():
            yield key

    def words(self):
        """ Iterator over the words in the dictionary
            Yields:
                str: The next word in the dictionary
          """
        for word in self._dictionary.keys():
            yield word

    def items(self):
        """ Iterator over the words in the dictionary
            Yields:
                str: The next word in the dictionary
                int: The number of instances in the dictionary
            Note:
                This is the same as `dict.items()` """
        for word in self._dictionary.keys():
            yield word, self._dictionary[word]

    def load_dictionary(self, filename):
        """ Load in a pre-built word frequency list
            Args:
                filename (str): The filepath to the json (optionally gzipped) \
                file to be loaded """
        self._dictionary.update(pickle.load(open(filename, 'rb')))
        self._update_dictionary()

    def remove_by_threshold(self, threshold=5):
        """ Remove all words at, or below, the provided threshold
            Args:
                threshold (int): The threshold at which a word is to be removed """
        keys = [x for x in self._dictionary.keys()]
        for key in keys:
            if self._dictionary[key] <= threshold:
                self._dictionary.pop(key)
        self._update_dictionary()

    def _update_dictionary(self):
        """ Update the word frequency object """
        self._longest_word_length = 0
        self._total_words = sum(self._dictionary.values())
        self._unique_words = len(self._dictionary.keys())
        self._letters = set()
        for key in self._dictionary:
            if len(key[0]) > self._longest_word_length:
                self._longest_word_length = len(key[0])
                self._longest_word = key[0]
        self._letters.update(string.ascii_lowercase)

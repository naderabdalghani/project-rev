
from utils import load_file
from collections import Counter
import json


def export_word_frequency(filepath, word_frequency):
    """ Generate a json object from a word frequency
        Args:
            filepath (str):
            word_frequency (Counter):
    """
    with open(filepath, 'w') as f:
        json.dump(word_frequency, f, indent="", sort_keys=True, ensure_ascii=False)


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def build_word_frequency(filepath, output_path):
    """ Parse the passed text file (from Open Subtitles) into
        a word frequency list and save it to disk
        Args:
            filepath (str):
            output_path (str):
        Returns:
            Counter: The word frequency as parsed from the file
        Note:
            This only removes words that are proper nouns (attempts to...) and
            anything that starts or stops with something that is not in the alphabet.
    """
    # NLTK is only needed in this portion of the project
    try:
        from nltk.tag import pos_tag
        from nltk.tokenize import WhitespaceTokenizer
    except ImportError as ex:
        raise ImportError("To build a dictionary from scratch, NLTK is required!\n{}".format(ex.message))

    word_frequency = Counter()
    tok = WhitespaceTokenizer()
    idx = 0
    vowels = set("aeiouy")

    # num_lines = sum(1 for line in open(filepath))
    # num_lines = 441450449
    with load_file(filepath) as fobj:
        for line in fobj:
            # print(line)
            line = line.lower()

            # tokenize into parts
            parts = tok.tokenize(line)

            # Attempt to remove proper nouns
            tagged_sent = pos_tag(parts)

            # Remove words with invalid characters
            # Remove things that have leading or trailing non-alphabetic characters.
            # Remove words without a vowel
            # Remove ellipses
            # Remove Double punctuations
            words = [word[0] for word in tagged_sent if word[0] and is_english(word[0]) and not word[1] == "NNP" and word[0].isalnum() and not vowels.isdisjoint(word[0]) and not word[0].count("'") > 1 and not word[0].count(".") > 2 and not ".." in word[0] and word[0][0].isalpha() and word[0][-1].isalpha()]

            if words:
                # print(words)
                word_frequency.update(words)

            idx += 1

            if idx % 100000 == 0:
                print("completed: {} rows".format(idx))

    print("completed: {} rows".format(idx))

    export_word_frequency(output_path, word_frequency)

    return word_frequency

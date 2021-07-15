import json
import word_frequency
import string
import os
from utils import _parse_into_words, write_file
from utilities.config import OUTPUT_DIR


class LanguageModel(object):
    """ The Auto correction model class encapsulates the basics needed to accomplish a
        simple spell checking algorithm. """

    def __init__(
        self,
        local_dictionary=None,
    ):

        self._tokenizer = _parse_into_words
        self._word_frequency = word_frequency.WordFrequency()

        if local_dictionary:
            self._word_frequency.load_dictionary(local_dictionary)
        else:
            raise Exception("Sorry, There is no Dictionary to load Please enter the path of the dictionary")

    def __contains__(self, key):
        """ setup easier known checks """
        return key in self._word_frequency

    def __getitem__(self, key):
        """ setup easier frequency checks """
        return self._word_frequency[key]

    def __iter__(self):
        """ setup iter support """
        for word in self._word_frequency.dictionary:
            yield word

    @property
    def word_frequency(self):
        """ WordFrequency: An encapsulation of the word frequency `dictionary`
            Note:
                Not settable """
        return self._word_frequency

    def split_words(self, text):
        """ Split text into individual `words` using either a simple whitespace
            regex or the passed in tokenizer
            Args:
                text (str): The text to split into individual words
            Returns:
                list(str): A listing of all words in the provided text """
        return self._tokenizer(text)

    def export(self, filepath, gzipped=True):
        """ Export the word frequency list for import in the future
             Args:
                filepath (str): The filepath to the exported dictionary
                gzipped (bool): Whether to gzip the dictionary or not """
        data = json.dumps(self.word_frequency.dictionary, sort_keys=True)
        write_file(filepath, gzipped, data)

    def word_probability(self, word):
        """ Calculate the frequency to the `word` provided as seen across the
            entire dictionary
            Args:
                word (str): The word for which the word probability is calculated
            Returns:
                float: The probability that the word is the correct word """

        total_words = self._word_frequency.total_words
        return self._word_frequency.dictionary[word] / total_words

    def get_correction(self, word):
        """ The most probable correct spelling for the word
            Args:
                word (str): The word to correct
            Returns:
                str: The most likely correction """
        corrections = list(self.get_corrections(word))
        return max(sorted(corrections), key=self.__getitem__)

    def get_corrections(self, word):
        """ Generate possible spelling corrections for the provided word up to
            an edit distance of two, if and only when needed
            Args:
                word (str): The word for which to calculate candidate spellings
            Returns:
                set: The set of words that are possible corrections """
        if self.known([word]):  # short-cut if word is correct already
            return {word}

        if not self.should_check(word):
            return {word}

        # get edit distance 1...
        res = [x for x in self.edit_one_letter(word)]
        tmp = self.known(res)
        if tmp:
            return tmp
        # if still not found, use the edit distance 1 to calc edit distance 2
        else:
            tmp = self.known([x for x in self.edit_two_letters(res)])
            if tmp:
                return tmp
        return {word}

    def known(self, words):
        """ The subset of `words` that appear in the dictionary of words
            Args:
                words (list): List of words to determine which are in the corpus
            Returns:
                set: The set of those words from the input that are in the \
                corpus """
        words = [w for w in words]
        tmp = [w.lower() for w in words]
        return set(
            w
            for w in tmp
            if w in self._word_frequency.dictionary and self.should_check(w)
        )

    def edit_one_letter(self, word):
        """ Compute all strings that are one edit away from `word` using only
            the letters in the corpus
            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance one from the \
                provided word """
        word = (
            word.lower()
        )
        if self.should_check(word) is False:
            return {word}
        letters = self._word_frequency.letters
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def should_check(self, word):
        """ Check if should i check that word or not
             Args:
                 word: The word for which to check
             Returns:
                 bool: true if yes false if i shouldn't check """
        if len(word) == 1 and word in string.punctuation:
            return False
        if (
            len(word) > self._word_frequency.longest_word_length + 3  # 2 or 3 ?
        ):  # magic number to allow removal of up to 2 letters.
            return False
        try:  # check if it is a number (int, float, etc)
            float(word)
            return False
        except ValueError:
            pass

        return True

    def edit_two_letters(self, words):
        """ Compute all strings that are 2 edits away from all the words
            Args:
                words (list): The words for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance two from the \
                provided words """
        words = [w for w in words]
        tmp = [
            w.lower()
            for w in words
            if self.should_check(w)
        ]
        output = [e2 for e1 in tmp for e2 in self.known(self.edit_one_letter(e1))]
        return set(output)

    def auto_correction_model(self,text):
        """ Correct the given sentence
            Args:
                text (string): The sentence for which to correct it
            Returns:
                text: The corrected sentence """
        corrected_sentence = ""
        words = self.split_words(text)
        for word in words:
            corrected_sentence += self.get_correction(word) + " "
        return corrected_sentence


if __name__ == '__main__':

    # Code of merging dictionaries
    # word_frequency = Counter()
    # filename = "D:\\cmp\\4th Year\\GP\\Auto_correction_model\\New folder\\output"
    # for i in range(1, 6, 1):
    #     with open(filename + str(i), "r") as data:
    #         data = data.read()
    #         data = data.lower()
    #         word_frequency.update(json.loads(data))
    #
    # export_word_frequency("D:\\cmp\\4th Year\\GP\\Auto_correction_model\\New folder\\dictionary", word_frequency)

    ACM = LanguageModel(os.path.join(OUTPUT_DIR, "dictionary"))
    # ACM._word_frequency.remove_by_threshold(threshold=15)
    correct = ACM.auto_correction_model("")
    print(correct)

import json
import word_frequency
import os
import pickle
import numpy as np
from utils import parse_into_words, write_file
from app_config import MODELS_DIR

loaded_language_model = None


def load_language_model():
    pass


class LanguageModel:
    """ The Auto correction model class encapsulates the basics needed to accomplish a
        simple spell checking algorithm. """

    def __init__(
            self,
            local_dictionary=None,
    ):

        self._tokenizer = parse_into_words
        self._word_frequency = word_frequency.WordFrequency()

        if local_dictionary:
            self._word_frequency.load_dictionary(os.path.join(MODELS_DIR, "unigrams_tuples"))
            # self._word_frequency.remove_by_threshold(5)
            self._bi_grams = pickle.load(open(os.path.join(MODELS_DIR, "bigrams_tuples"), 'rb'))
            self._tri_grams = pickle.load(open(os.path.join(MODELS_DIR, "trigrams_tuples"), 'rb'))
            self._names = pickle.load(open(os.path.join(MODELS_DIR, "names"), 'rb'))
            self._uni_grams_size = self._word_frequency.unique_words
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

    def get_correction(self, words, word, i):
        """ The most probable correct spelling for the word
            Args:
                words (list): List of tokenized words
                word (str): The word to correct
                i (str): Index of the word to be processed
            Returns:
                str: The most likely correction """
        probabilities_arr = []
        max_index = 0
        sentence = np.copy(words)
        corrections = list(self.get_corrections(word))
        old_word_idx = 0
        if len(corrections) > 1:
            for j, correction in enumerate(corrections):
                sentence[i] = correction
                probabilities_arr.append(self.estimate_sentence_probability(sentence))
                if correction == word:
                    old_word_idx = j
            maximum = max(probabilities_arr)
            max_index = probabilities_arr.index(maximum)
            if maximum == probabilities_arr[old_word_idx]:
                return corrections[old_word_idx]
        return corrections[max_index]

    def get_corrections(self, word):
        """ Generate possible spelling corrections for the provided word up to
            an edit distance of two, if and only when needed
            Args:
                word (str): The word for which to calculate candidate spellings
            Returns:
                set: The set of words that are possible corrections """
        if self.known([word]) and not self.should_check(word):  # short-cut if word is correct already
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
            if (w,) in self._word_frequency.dictionary
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

        try:  # check if it is a number (int, float, etc)
            float(word)
            return False
        except ValueError:
            pass

        if (
                len(word) > self._word_frequency.longest_word_length + 3  # 2 or 3 ?
        ):  # magic number to allow removal of up to 2 letters.
            return False

        if len(word) == 1:
            return False
        return True

        # if len(word) == 1 and word in string.punctuation:
        #     return False

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

    def auto_correction_model(self, text):
        """ Correct the given sentence
            Args:
                text (string): The sentence for which to correct it
            Returns:
                text: The corrected sentence """
        corrected_sentence = ""
        text = text.lower()
        words = self.split_words(text)
        for i, word in enumerate(words):
            if word.title() not in self._names:
                correct = self.get_correction(words, word, i)
            else:
                correct = word
            corrected_sentence += correct + " "
            words[i] = correct
        return corrected_sentence

    def estimate_probability(self, word, previous_n_gram, tri=True, k=1.0):
        """
        Estimate the probabilities of a next word using the n-gram counts with k-smoothing

        Args:
            word: next word
            previous_n_gram: A sequence of words of length n
            tri: Whether to use trigrams or not
            k: positive constant, smoothing parameter

        Returns:
            A probability
        """
        # convert list to tuple to use it as a dictionary key
        if isinstance(previous_n_gram, str):
            previous_n_gram = (previous_n_gram,)
        else:
            previous_n_gram = tuple(previous_n_gram)

        # Set the denominator
        # If the previous n-gram exists in the dictionary of n-gram counts,
        # Get its count.  Otherwise set the count to zero
        # Use the dictionary that has counts for n-grams
        if not tri:
            previous_n_gram_count = self._word_frequency.dictionary.get(previous_n_gram, 0)
        else:
            previous_n_gram_count = self._bi_grams.get(previous_n_gram, 0)

        # Calculate the denominator using the count of the previous n gram
        # and apply k-smoothing
        denominator = previous_n_gram_count + k * self._uni_grams_size

        # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
        n_plus1_gram = previous_n_gram + (word,)

        # Set the count to the count in the dictionary,
        # otherwise 0 if not in the dictionary
        # use the dictionary that has counts for the n-gram plus current word
        if not tri:
            n_plus1_gram_count = self._bi_grams.get(n_plus1_gram, 0)
        else:
            n_plus1_gram_count = self._tri_grams.get(n_plus1_gram, 0)

        # Define the numerator use the count of the n-gram plus current word,
        # and apply smoothing
        numerator = n_plus1_gram_count + k

        # Calculate the probability as the numerator divided by denominator
        probability = numerator / denominator

        return probability

    def estimate_sentence_probability(self, sentence, tri=True, k=1.0):
        sentence_to_check = np.copy(sentence)
        sentence_to_check = np.insert(sentence_to_check, 0, "<s>", axis=0)
        if tri:
            sentence_to_check = np.insert(sentence_to_check, 0, "<s>", axis=0)
        sentence_to_check = np.insert(sentence_to_check, len(sentence_to_check), "<e>", axis=0)
        prob = 0.0
        for i, word in enumerate(sentence_to_check):
            if i == len(sentence_to_check) - 1 and not tri:
                return prob
            if i == len(sentence_to_check) - 2 and tri:
                return prob
            if not tri:
                prob1 = self.estimate_probability(sentence_to_check[i + 1], sentence_to_check[i], tri, k)
            else:
                prob1 = self.estimate_probability(sentence_to_check[i + 2], sentence_to_check[i:i + 2], tri, k)
            prob = prob + np.log(prob1)

    def calculate_perplexity(self, test_data_processed):
        perplexity = 0.0
        for i, sentence in enumerate(test_data_processed):
            perplexity += self.estimate_sentence_probability(sentence, tri=True, k=.001)
            if i % 100000 == 0:
                print(str(i) + " sentences are completed")
        perplexity = perplexity / float(len(test_data_processed))
        print(perplexity)


def main():
    loaded = True
    ACM = LanguageModel(loaded)
    # ACM._word_frequency.remove_by_threshold(threshold=15)
    # correct = ACM.auto_correction_model("I am adpicted to foutbull")
    # p1 = ACM.estimate_sentence_probability(["how", "are", "you", "doing"])
    # p2 = ACM.estimate_sentence_probability(["how", "are", "you", "doink"])
    corrections = []
    correct = ACM.auto_correction_model("hellow ted how is it goink")
    corrections.append(correct)
    correct = ACM.auto_correction_model("I am adpicted to foutbull")
    corrections.append(correct)
    correct = ACM.auto_correction_model("how ar youu doink")
    corrections.append(correct)
    correct = ACM.auto_correction_model("i red a book")
    corrections.append(correct)
    correct = ACM.auto_correction_model("i wont a cake")
    corrections.append(correct)
    correct = ACM.auto_correction_model("i like football and basketball")
    corrections.append(correct)
    correct = ACM.auto_correction_model("do you now robin")
    corrections.append(correct)
    x = 1


if __name__ == '__main__':
    main()

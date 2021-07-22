import re
import gzip
import contextlib


def _parse_into_words(text):
    """ Parse the text into words; currently removes punctuation except for
        apostrophies.
        Args:
            text (str): The text to split into words
    """
    return re.findall(r"(\w[\w']*\w|\w)", text)


@contextlib.contextmanager
def load_file(filename):
    if filename[-3:].lower() == ".gz":
        with gzip.open(filename, mode="rt", encoding="utf-8") as fobj:
            yield fobj
    else:
        with open(filename, mode="r", encoding="utf-8") as fobj:
            yield fobj


def write_file(filepath, gzipped, data):
    """ Write the data to file either as a gzip file or text based on the
        gzipped parameter
        Args:
            filepath (str): The filename to open
            gzipped (bool): Whether the file should be gzipped or not
            data (str): The data to be written out
    """
    if gzipped:
        with gzip.open(filepath, "wt") as fobj:
            fobj.write(data)
    else:
        with open(filepath, "w", encoding='utf-8') as fobj:
            fobj.write(data)
